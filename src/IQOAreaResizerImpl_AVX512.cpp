#include "IQOAreaResizerImpl.hpp"


#if defined(IQO_CPU_X86) && defined(IQO_HAVE_AVX512)

#include <cstring>
#include <vector>
#include <immintrin.h>

#if defined(_OPENMP)
    #include <omp.h>
#endif

#include "IQOHWCap.hpp"


namespace {

    int getNumberOfProcs()
    {
#if defined(_OPENMP)
        return omp_get_num_procs();
#else
        return 1;
#endif
    }

    int getThreadNumber()
    {
#if defined(_OPENMP)
        return omp_get_thread_num();
#else
        return 0;
#endif
    }

    uint8_t cvt_roundss_su8(float v)
    {
        __m128  f32x1V     = _mm_set_ss(v);
        __m128  f32x1Round = _mm_round_ss(f32x1V, f32x1V, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        int32_t s32x1Round = _mm_cvtss_si32(f32x1Round);
        return clamp(0, 255, s32x1Round);
    }

    //! (uint8_t)min(255, max(0, round(v)))
    __m256i cvt_roundps_epu8(__m512 lo, __m512 hi)
    {
        __m512i s32x16L = _mm512_cvt_roundps_epi32(lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512i s32x16H = _mm512_cvt_roundps_epi32(hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // HFHEHDHC LFLELDLC HBHAH9H8 LBLAL9L8 H7H6H5H4 L7L6L5L4 H3H2H1H0 L3L2L1L0
        __m512i u16x32P  = _mm512_packus_epi32(s32x16L, s32x16H);
        __m256i u16x16P0 = _mm512_castsi512_si256(u16x32P);
        __m256i u16x16P1 = _mm512_extracti64x4_epi64(u16x32P, 1);
        // HFHEHDHC LFLELDLC H7H6H5H4 L7L6L5L4 HBHAH9H8 LBLAL9L8 H3H2H1H0 L3L2L1L0
        __m256i u8x32P = _mm256_packus_epi16(u16x16P0, u16x16P1);
        __m256i u32x8Table = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);
        __m256i u8x32  = _mm256_permutexvar_epi32(u32x8Table, u8x32P);
        return u8x32;
    }

}

namespace iqo {

    template<>
    class AreaResizerImpl<ArchAVX512> : public IAreaResizerImpl
    {
    public:
        //! Destructor
        virtual ~AreaResizerImpl() {}

        //! Construct
        virtual void init(
            size_t srcW, size_t srcH,
            size_t dstW, size_t dstH
        );

        //! Run image resizing
        virtual void resize(
            size_t srcSt, const unsigned char * src,
            size_t dstSt, unsigned char * dst
        );

    private:
        void resizeYmain(
            ptrdiff_t srcSt, const uint8_t * src,
            ptrdiff_t dstW, float * dst,
            ptrdiff_t srcOY,
            const float * coefs);
        void resizeX(const float * src, uint8_t * dst);
        void resizeXmain(const float * src, uint8_t * dst);

        enum {
            //! for SIMD
            kVecStepY  = 32, //!< __m512 x 2
            kVecStepX  = 32, //!< __m512 x 2
        };
        ptrdiff_t m_SrcW;
        ptrdiff_t m_SrcH;
        ptrdiff_t m_DstW;
        ptrdiff_t m_DstH;
        ptrdiff_t m_NumCoefsX;
        ptrdiff_t m_NumCoefsY;
        ptrdiff_t m_NumCoordsX;
        ptrdiff_t m_NumUnrolledCoordsX;
        ptrdiff_t m_TablesXWidth;
        ptrdiff_t m_NumCoordsY;
        std::vector<float> m_TablesX_;  //!< Area table * m_NumCoordsX (unrolled)
        float * m_TablesX;              //!< aligned
        std::vector<float> m_TablesY;   //!< Area table * m_NumCoordsY
        std::vector<int32_t> m_IndicesX;
        std::vector<float> m_Work;
    };

    template<>
    bool AreaResizerImpl_hasFeature<ArchAVX512>()
    {
        HWCap cap;
        return cap.hasAVX512F()
            && cap.hasAVX512VL()
            && cap.hasAVX512BW()
            && cap.hasAVX512DQ()
            && cap.hasAVX512CD();
    }

    template<>
    IAreaResizerImpl * AreaResizerImpl_new<ArchAVX512>()
    {
        return new AreaResizerImpl<ArchAVX512>();
    }


    // Constructor
    void AreaResizerImpl<ArchAVX512>::init(
        size_t srcW, size_t srcH,
        size_t dstW, size_t dstH
    ) {
        m_SrcW = srcW;
        m_SrcH = srcH;
        m_DstW = dstW;
        m_DstH = dstH;

        // setup coefficients
        size_t gcdW = gcd(m_SrcW, m_DstW);
        size_t gcdH = gcd(m_SrcH, m_DstH);
        size_t rSrcW = m_SrcW / gcdW;
        size_t rDstW = m_DstW / gcdW;
        size_t rSrcH = m_SrcH / gcdH;
        size_t rDstH = m_DstH / gcdH;
        m_NumCoefsX = calcNumCoefsForArea(rSrcW, rDstW);
        m_NumCoefsY = calcNumCoefsForArea(rSrcH, rDstH);
        m_NumCoordsX = rDstW;
        m_NumUnrolledCoordsX = std::min(alignCeil(m_DstW, kVecStepX), lcm(m_NumCoordsX, kVecStepX));
        m_NumUnrolledCoordsX /= kVecStepX;
        m_TablesXWidth = kVecStepX * m_NumCoefsX;
        m_NumCoordsY = rDstH;
        m_TablesX_.reserve(m_TablesXWidth * m_NumUnrolledCoordsX + kVecStepX);
        m_TablesX_.resize(m_TablesXWidth * m_NumUnrolledCoordsX + kVecStepX);
        m_TablesX = (float *)alignCeil(intptr_t(&m_TablesX_[0]), sizeof(*m_TablesX) * kVecStepX);
        m_TablesY.reserve(m_NumCoefsY * m_NumCoordsY);
        m_TablesY.resize(m_NumCoefsY * m_NumCoordsY);

        std::vector<float> tablesX(m_NumCoefsX * m_NumCoordsX);
        for ( ptrdiff_t dstX = 0; dstX < m_NumCoordsX; ++dstX ) {
            float * table = &tablesX[dstX * m_NumCoefsX];
            float sumCoefs = setAreaTable(rSrcW, rDstW, dstX, m_NumCoefsX, table);
            for ( ptrdiff_t i = 0; i < m_NumCoefsX; ++i ) {
                table[i] /= sumCoefs;
            }
        }

        // unroll X coefs
        //
        //      srcX: A B C    (upto m_NumCoordsX)
        //    coef #: 0 1 2 3
        //   tablesX: A0A1A2A3
        //            B0B1B2B3
        //            C0C1C2C3
        //
        //      srcX: ABCA BCAB CABC (upto m_NumUnrolledCoordsX; unroll to kVecStepX)
        // m_TablesX: A0B0C0A0 .. A3B3C3A3
        //            B0C0A0B0 .. B3C3A3B3
        //            C0A0B0C0 .. C3A3B3C3
        //
        // other cases
        //
        //      srcX: A B         -> ABAB
        //      srcX: A B C D E   -> ABCD EABC DEAB CDEA BCDE
        //      srcX: A B C D E F -> ABCD EFAB CDEF
        ptrdiff_t nCoefs  = m_NumCoefsX;
        ptrdiff_t nCoords = m_NumCoordsX;
        ptrdiff_t nCols   = m_TablesXWidth;
        ptrdiff_t nRows   = m_NumUnrolledCoordsX;
        for ( ptrdiff_t row = 0; row < nRows; ++row ) {
            for ( ptrdiff_t col = 0; col < nCols; ++col ) {
                ptrdiff_t iCoef = col/kVecStepX;
                ptrdiff_t srcX = (col%kVecStepX + row*kVecStepX) % nCoords;
                m_TablesX[col + nCols*row] = tablesX[iCoef + srcX*nCoefs];
            }
        }

        for ( ptrdiff_t dstY = 0; dstY < m_NumCoordsY; ++dstY ) {
            float * table = &m_TablesY[dstY * m_NumCoefsY];
            float sumCoefs = setAreaTable(rSrcH, rDstH, dstY, m_NumCoefsY, table);
            for ( ptrdiff_t i = 0; i < m_NumCoefsY; ++i ) {
                table[i] /= sumCoefs;
            }
        }

        // allocate workspace
        m_Work.reserve(m_SrcW * getNumberOfProcs());
        m_Work.resize(m_SrcW * getNumberOfProcs());

        // calc indices
        ptrdiff_t alignedDstW = alignCeil(m_DstW, kVecStepX);
        m_IndicesX.reserve(alignedDstW);
        m_IndicesX.resize(alignedDstW);
        for ( ptrdiff_t dstX = 0; dstX < alignedDstW; ++dstX ) {
            //      srcOX = floor(dstX / scale)
            int32_t srcOX = int32_t(dstX * rSrcW / rDstW);
            m_IndicesX[dstX] = srcOX;
        }
    }

    void AreaResizerImpl<ArchAVX512>::resize(
        size_t srcSt, const uint8_t * src,
        size_t dstSt, uint8_t * __restrict dst
    ) {
        if ( m_SrcH == m_DstH ) {
#pragma omp parallel for
            for ( ptrdiff_t y = 0; y < m_SrcH; ++y ) {
                float * work = &m_Work[getThreadNumber() * m_SrcW];
                for ( ptrdiff_t x = 0; x < m_SrcW; ++x ) {
                    work[x] = src[srcSt * y + x];
                }
                resizeX(work, &dst[dstSt * y]);
            }
            return;
        }

        const float * tablesY = &m_TablesY[0];
        ptrdiff_t dstH = m_DstH;

        // main loop
#pragma omp parallel for
        for ( ptrdiff_t dstY = 0; dstY < dstH; ++dstY ) {
            float * work = &m_Work[getThreadNumber() * m_SrcW];
            ptrdiff_t srcOY = dstY * m_SrcH / m_DstH;
            const float * coefs = &tablesY[dstY % m_NumCoordsY * m_NumCoefsY];
            resizeYmain(
                srcSt, &src[0],
                m_SrcW, work,
                srcOY,
                coefs);
            resizeX(work, &dst[dstSt * dstY]);
        }
    }

    //! resize vertical (main loop)
    //!
    //! @param srcSt  Stride in src (in byte)
    //! @param src    A row of source
    //! @param dst    A row of destination
    //! @param srcOY  The origin of current line
    //! @param coefs  The coefficients
    void AreaResizerImpl<ArchAVX512>::resizeYmain(
        ptrdiff_t srcSt, const uint8_t * src,
        ptrdiff_t dstW, float * __restrict dst,
        ptrdiff_t srcOY,
        const float * coefs
    ) {
        ptrdiff_t numCoefsY = m_NumCoefsY;
        ptrdiff_t vecLen = alignFloor(dstW, kVecStepY);

        for ( ptrdiff_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            // nume = 0;
            __m512 f32x16Nume0 = _mm512_setzero_ps();
            __m512 f32x16Nume1 = _mm512_setzero_ps();

            for ( ptrdiff_t i = 0; i < numCoefsY; ++i ) {
                ptrdiff_t srcY = srcOY + i;
                // coef = coefs[i];
                __m512  f32x16Coef = _mm512_set1_ps(coefs[i]);
                // nume += src[dstX + srcSt*srcY] * coef;
                __m256i u8x32Src    = _mm256_loadu_si256((const __m256i *)&src[dstX + srcSt*srcY]);
                __m128i u8x16Src0   = _mm256_castsi256_si128(u8x32Src);
                __m128i u8x16Src1   = _mm256_extracti128_si256(u8x32Src, 1);
                __m512  f32x16Src0  = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u8x16Src0));
                __m512  f32x16Src1  = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u8x16Src1));
                f32x16Nume0 = _mm512_fmadd_ps(f32x16Src0, f32x16Coef, f32x16Nume0);
                f32x16Nume1 = _mm512_fmadd_ps(f32x16Src1, f32x16Coef, f32x16Nume1);
            }

            // dst[dstX] = nume;
            __m512 f32x16Dst0 = f32x16Nume0;
            __m512 f32x16Dst1 = f32x16Nume1;
            _mm512_storeu_ps(&dst[dstX +  0], f32x16Dst0);
            _mm512_storeu_ps(&dst[dstX + 16], f32x16Dst1);
        }

        for ( ptrdiff_t dstX = vecLen; dstX < dstW; dstX++ ) {
            float nume = 0;

            for ( ptrdiff_t i = 0; i < numCoefsY; ++i ) {
                ptrdiff_t srcY = srcOY + i;
                float    coef = coefs[i];
                nume += src[dstX + srcSt * srcY] * coef;
            }

            dst[dstX] = nume;
        }
    }

    void AreaResizerImpl<ArchAVX512>::resizeX(const float * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            ptrdiff_t dstW = m_DstW;
            for ( ptrdiff_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = cvt_roundss_su8(src[dstX]);
            }
            return;
        }

        resizeXmain(src, dst);
    }

    //! resize horizontal (main loop)
    //!
    //! @param src    A row of source
    //! @param dst    A row of destination
    void AreaResizerImpl<ArchAVX512>::resizeXmain(const float * src, uint8_t * __restrict dst)
    {
        const float * coefs = &m_TablesX[0];
        const int32_t * indices = &m_IndicesX[0];
        ptrdiff_t tableSize = m_TablesXWidth * m_NumUnrolledCoordsX;
        ptrdiff_t numCoefsX = m_NumCoefsX;
        ptrdiff_t dstW = m_DstW;
        ptrdiff_t vecLen = alignFloor(dstW, kVecStepX);

        ptrdiff_t iCoef = 0;
        for ( ptrdiff_t dstX = 0; dstX < vecLen; dstX += kVecStepX ) {
            //      nume   = 0;
            __m512  f32x16Nume0 = _mm512_setzero_ps();
            __m512  f32x16Nume1 = _mm512_setzero_ps();
            //      srcX = floor(dstX / scale) + i;
            __m512i s32x16SrcOX0 = _mm512_loadu_si512((const __m512i*)&indices[dstX +  0]);
            __m512i s32x16SrcOX1 = _mm512_loadu_si512((const __m512i*)&indices[dstX + 16]);

            for ( ptrdiff_t i = 0; i < numCoefsX; ++i ) {
                //ptrdiff_t srcX = indices[dstX] + offset;
                __m512i s32x16Offset   = _mm512_set1_epi32(i);
                __m512i s32x16SrcX0 = _mm512_add_epi32(s32x16SrcOX0, s32x16Offset);
                __m512i s32x16SrcX1 = _mm512_add_epi32(s32x16SrcOX1, s32x16Offset);

                //nume[dstX] += src[srcX] * coefs[iCoef];
                __m512  s32x16Src0    = _mm512_i32gather_ps(s32x16SrcX0, src, sizeof(float));
                __m512  s32x16Src1    = _mm512_i32gather_ps(s32x16SrcX1, src, sizeof(float));
                __m512  f32x16Coefs0  = _mm512_load_ps(&coefs[iCoef +  0]);
                __m512  f32x16Coefs1  = _mm512_load_ps(&coefs[iCoef + 16]);
                f32x16Nume0 = _mm512_fmadd_ps(s32x16Src0, f32x16Coefs0, f32x16Nume0);
                f32x16Nume1 = _mm512_fmadd_ps(s32x16Src1, f32x16Coefs1, f32x16Nume1);

                iCoef += kVecStepX;
            }

            __m256i u8x32Dst = cvt_roundps_epu8(f32x16Nume0, f32x16Nume1);
            _mm256_storeu_si256((__m256i*)&dst[dstX], u8x32Dst);

            // iCoef = dstX % tableSize;
            if ( iCoef == tableSize ) {
                iCoef = 0;
            }
        }

        for ( ptrdiff_t dstX = vecLen; dstX < dstW; ++dstX ) {
            ptrdiff_t srcOX = indices[dstX];
            float sum = 0;

            // calc index of coefs from unrolled table
            iCoef = (dstX % kVecStepX) + (dstX / kVecStepX % m_NumUnrolledCoordsX * m_TablesXWidth);

            for ( ptrdiff_t i = 0; i < numCoefsX; ++i ) {
                ptrdiff_t srcX = srcOX + i;
                sum += src[srcX] * coefs[iCoef];
                iCoef += kVecStepX;
            }

            dst[dstX] = cvt_roundss_su8(sum);
        }
    }

}

#else

namespace iqo {

    template<>
    bool AreaResizerImpl_hasFeature<ArchAVX512>()
    {
        return false;
    }

    template<>
    IAreaResizerImpl * AreaResizerImpl_new<ArchAVX512>()
    {
        return NULL;
    }

}

#endif
