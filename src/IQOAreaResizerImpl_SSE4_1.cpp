#include "IQOAreaResizerImpl.hpp"


#if defined(IQO_CPU_X86) && defined(IQO_HAVE_SSE4_1)

#include <cstring>
#include <vector>
#include <smmintrin.h>

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

    //! f32x4Dst[dstField] = srcPtr[s32x4Indices[srcField]]
    #define IQO_INSERT_MEM_PS(srcPtr, s32x4Indices, srcField, f32x4Dst, dstField) \
        _mm_insert_ps( \
            (f32x4Dst), \
            _mm_load_ss(&(srcPtr)[_mm_extract_epi32((s32x4Indices), (srcField))]), \
            (dstField) << 4 \
        )

    __m128 gather_ps(const float * f32Src, __m128i s32x4Indices)
    {
        __m128 f32x4V = _mm_setzero_ps();
        f32x4V = IQO_INSERT_MEM_PS(f32Src, s32x4Indices, 0, f32x4V, 0);
        f32x4V = IQO_INSERT_MEM_PS(f32Src, s32x4Indices, 1, f32x4V, 1);
        f32x4V = IQO_INSERT_MEM_PS(f32Src, s32x4Indices, 2, f32x4V, 2);
        f32x4V = IQO_INSERT_MEM_PS(f32Src, s32x4Indices, 3, f32x4V, 3);
        return f32x4V;
    }

    uint8_t cvt_roundss_su8(float v)
    {
        __m128  f32x1V     = _mm_set_ss(v);
        __m128  f32x1Round = _mm_round_ss(f32x1V, f32x1V, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        int32_t s32x1Round = _mm_cvtss_si32(f32x1Round);
        return clamp(0, 255, s32x1Round);
    }

    __m128i cvt_roundps_epu8(__m128 lo, __m128 hi)
    {
        __m128  f32x4Round0 = _mm_round_ps(lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128  f32x4Round1 = _mm_round_ps(hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i s32x4Round0 = _mm_cvtps_epi32(f32x4Round0);
        __m128i s32x4Round1 = _mm_cvtps_epi32(f32x4Round1);
        __m128i u16x8Round  = _mm_packus_epi32(s32x4Round0, s32x4Round1);
        return _mm_packus_epi16(u16x8Round, u16x8Round);
    }

}

namespace iqo {

    template<>
    class AreaResizerImpl<ArchSSE4_1> : public IAreaResizerImpl
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
            const float * coefs
        );
        void resizeX(const float * src, uint8_t * dst);
        void resizeXmain(const float * src, uint8_t * dst);

        enum {
            //! for SIMD
            kVecStepX  =  8, //!< float32x4 x 2
            kVecStepY  = 16, //!< float32x4 x 4
        };
        ptrdiff_t m_SrcW;
        ptrdiff_t m_SrcH;
        ptrdiff_t m_DstW;
        ptrdiff_t m_DstH;
        int32_t   m_NumCoefsX;
        ptrdiff_t m_NumCoefsY;
        ptrdiff_t m_NumCoordsX;
        ptrdiff_t m_NumUnrolledCoordsX;
        ptrdiff_t m_TablesXWidth;
        ptrdiff_t m_NumCoordsY;
        std::vector<float > m_TablesX_; //!< m_TablesXWidth * m_NumCoordsX (unrolled)
        float * m_TablesX;              //!< aligned
        std::vector<float> m_TablesY;   //!< Area table * m_NumCoordsY
        std::vector<float> m_Work;
        std::vector<int32_t> m_IndicesX;
    };

    template<>
    bool AreaResizerImpl_hasFeature<ArchSSE4_1>()
    {
        HWCap cap;
        return cap.hasSSE4_1();
    }

    template<>
    IAreaResizerImpl * AreaResizerImpl_new<ArchSSE4_1>()
    {
        return new AreaResizerImpl<ArchSSE4_1>();
    }


    // Constructor
    void AreaResizerImpl<ArchSSE4_1>::init(
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
        m_NumCoefsX = int32_t(calcNumCoefsForArea(rSrcW, rDstW));
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

        // X coefs
        std::vector<float> tablesX(m_NumCoefsX * m_NumCoordsX);
        for ( ptrdiff_t dstX = 0; dstX < m_NumCoordsX; ++dstX ) {
            float * table = &tablesX[dstX * m_NumCoefsX];
            float sumCoefs = setAreaTable(rSrcW, rDstW, dstX, m_NumCoefsX, table);
            for ( int i = 0; i < m_NumCoefsX; i++ ) {
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

        // Y coefs
        std::vector<float> tablesY(m_NumCoefsY);
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
            int32_t srcOX = int32_t(dstX * rSrcW / rDstW);
            m_IndicesX[dstX] = srcOX;
        }
    }

    void AreaResizerImpl<ArchSSE4_1>::resize(
        size_t srcSt, const uint8_t * src,
        size_t dstSt, uint8_t * __restrict dst
    ) {
        if ( m_SrcH == m_DstH ) {
#pragma omp parallel for
            for ( ptrdiff_t y = 0; y < m_SrcH; ++y ) {
                float * work = &m_Work[getThreadNumber() * m_SrcW];
                for ( ptrdiff_t x = 0; x < m_SrcW; ++x ) {
                    m_Work[m_SrcW * y + x] = src[srcSt * y + x];
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
    void AreaResizerImpl<ArchSSE4_1>::resizeYmain(
        ptrdiff_t srcSt, const uint8_t * src,
        ptrdiff_t dstW, float * __restrict dst,
        ptrdiff_t srcOY,
        const float * coefs
    ) {
        ptrdiff_t vecLen = alignFloor(dstW, kVecStepY);
        ptrdiff_t numCoefsY = m_NumCoefsY;

        for ( ptrdiff_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            // nume = 0;
            __m128 f32x4Nume0 = _mm_setzero_ps();
            __m128 f32x4Nume1 = _mm_setzero_ps();
            __m128 f32x4Nume2 = _mm_setzero_ps();
            __m128 f32x4Nume3 = _mm_setzero_ps();

            for ( ptrdiff_t i = 0; i < numCoefsY; ++i ) {
                ptrdiff_t srcY = srcOY + i;
                // coef = coefs[i];
                __m128  f32x4Coef   = _mm_set1_ps(coefs[i]);
                // nume += src[dstX + srcSt*srcY] * coef;
                __m128i u8x16Src    = _mm_loadu_si128((const __m128i *)&src[dstX + srcSt*srcY]);
                __m128i u8x4Src0    = u8x16Src;
                __m128i u8x4Src1    = _mm_shuffle_epi32(u8x16Src, _MM_SHUFFLE(3, 3, 3, 1));
                __m128i u8x4Src2    = _mm_shuffle_epi32(u8x16Src, _MM_SHUFFLE(3, 3, 3, 2));
                __m128i u8x4Src3    = _mm_shuffle_epi32(u8x16Src, _MM_SHUFFLE(3, 3, 3, 3));
                __m128  f32x4Src0   = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(u8x4Src0));
                __m128  f32x4Src1   = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(u8x4Src1));
                __m128  f32x4Src2   = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(u8x4Src2));
                __m128  f32x4Src3   = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(u8x4Src3));
                f32x4Nume0 = _mm_add_ps(_mm_mul_ps(f32x4Src0, f32x4Coef), f32x4Nume0);
                f32x4Nume1 = _mm_add_ps(_mm_mul_ps(f32x4Src1, f32x4Coef), f32x4Nume1);
                f32x4Nume2 = _mm_add_ps(_mm_mul_ps(f32x4Src2, f32x4Coef), f32x4Nume2);
                f32x4Nume3 = _mm_add_ps(_mm_mul_ps(f32x4Src3, f32x4Coef), f32x4Nume3);
            }

            // dst[dstX] = nume;
            __m128 f32x4Dst0 = f32x4Nume0;
            __m128 f32x4Dst1 = f32x4Nume1;
            __m128 f32x4Dst2 = f32x4Nume2;
            __m128 f32x4Dst3 = f32x4Nume3;
            _mm_storeu_ps(&dst[dstX +  0], f32x4Dst0);
            _mm_storeu_ps(&dst[dstX +  4], f32x4Dst1);
            _mm_storeu_ps(&dst[dstX +  8], f32x4Dst2);
            _mm_storeu_ps(&dst[dstX + 12], f32x4Dst3);
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

    void AreaResizerImpl<ArchSSE4_1>::resizeX(const float * src, uint8_t * __restrict dst)
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
    void AreaResizerImpl<ArchSSE4_1>::resizeXmain(const float * src, uint8_t * __restrict dst)
    {
        const float * coefs = &m_TablesX[0];
        const int32_t * indices = &m_IndicesX[0];
        ptrdiff_t tableSize = m_TablesXWidth * m_NumUnrolledCoordsX;
        int32_t numCoefsX = m_NumCoefsX;
        ptrdiff_t dstW = m_DstW;
        ptrdiff_t vecLen = alignFloor(dstW, kVecStepX);

        ptrdiff_t iCoef = 0;
        for ( ptrdiff_t dstX = 0; dstX < vecLen; dstX += kVecStepX ) {
            // nume             = 0;
            __m128 f32x4Nume0   = _mm_setzero_ps();
            __m128 f32x4Nume1   = _mm_setzero_ps();
            // srcOX            = floor(dstX / scale);
            __m128i s32x4SrcOX0 = _mm_loadu_si128((const __m128i*)&indices[dstX + 0]);
            __m128i s32x4SrcOX1 = _mm_loadu_si128((const __m128i*)&indices[dstX + 4]);

            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                // srcX             = srcOX + i;
                __m128i s32x4Offset = _mm_set1_epi32(i);
                __m128i s32x4SrcX0  = _mm_add_epi32(s32x4SrcOX0, s32x4Offset);
                __m128i s32x4SrcX1  = _mm_add_epi32(s32x4SrcOX1, s32x4Offset);
                // iNume            += src[srcX] * coefs[iCoef];
                __m128  f32x4Src0   = gather_ps(src, s32x4SrcX0);
                __m128  f32x4Src1   = gather_ps(src, s32x4SrcX1);
                __m128  f32x4Coefs0 = _mm_load_ps(&coefs[iCoef + 0]);
                __m128  f32x4Coefs1 = _mm_load_ps(&coefs[iCoef + 4]);
                __m128  f32x4iNume0 = _mm_mul_ps(f32x4Src0, f32x4Coefs0);
                __m128  f32x4iNume1 = _mm_mul_ps(f32x4Src1, f32x4Coefs1);
                // nume   += iNume;
                f32x4Nume0 = _mm_add_ps(f32x4Nume0, f32x4iNume0);
                f32x4Nume1 = _mm_add_ps(f32x4Nume1, f32x4iNume1);

                iCoef += kVecStepX;
            }

            // dst[dstX] = round(nume)
            __m128  f32x4Dst0     = f32x4Nume0;
            __m128  f32x4Dst1     = f32x4Nume1;
            __m128i u8x8Dst       = cvt_roundps_epu8(f32x4Dst0, f32x4Dst1);
            _mm_storel_epi64((__m128i*)&dst[dstX], u8x8Dst);

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
    bool AreaResizerImpl_hasFeature<ArchSSE4_1>()
    {
        return false;
    }

    template<>
    IAreaResizerImpl * AreaResizerImpl_new<ArchSSE4_1>()
    {
        return NULL;
    }

}

#endif
