#include <cstring>
#include <vector>
#include <immintrin.h>

#if defined(_OPENMP)
    #include <omp.h>
#endif

#include "IQOLanczosResizerImpl.hpp"
#include "IQOHWCap.hpp"

#if defined(IQO_CPU_X86) && defined(IQO_AVX2) && defined(IQO_FMA)

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
    class LanczosResizerImpl<ArchAVX512> : public ILanczosResizerImpl
    {
    public:
        //! Destructor
        virtual ~LanczosResizerImpl() {}

        //! Construct
        virtual void init(
            unsigned int degree,
            size_t srcW, size_t srcH,
            size_t dstW, size_t dstH,
            size_t pxScale
        );

        //! Run image resizing
        virtual void resize(
            size_t srcSt, const unsigned char * src,
            size_t dstSt, unsigned char * dst
        );

    private:
        void resizeYborder(
            intptr_t srcSt, const uint8_t * src,
            intptr_t dstW, float * dst,
            intptr_t srcOY,
            const float * coefs);
        void resizeYmain(
            intptr_t srcSt, const uint8_t * src,
            intptr_t dstW, float * dst,
            intptr_t srcOY,
            const float * coefs);
        void resizeX(const float * src, uint8_t * dst);
        void resizeXborder(
            const float * src, uint8_t * dst,
            intptr_t begin, intptr_t end);
        void resizeXmain(
            const float * src, uint8_t * dst,
            intptr_t begin, intptr_t end);

        enum {
            //! for SIMD
            kVecStepY  = 32, //!< __m512 x 2
            kVecStepX  = 32, //!< __m512 x 2
        };
        intptr_t m_SrcW;
        intptr_t m_SrcH;
        intptr_t m_DstW;
        intptr_t m_DstH;
        intptr_t m_NumCoefsX;
        intptr_t m_NumCoefsY;
        intptr_t m_NumCoordsX;
        intptr_t m_NumUnrolledCoordsX;
        intptr_t m_TablesXWidth;
        intptr_t m_NumCoordsY;
        std::vector<float> m_TablesX_;  //!< Lanczos table * m_NumCoordsX (unrolled)
        float * m_TablesX;              //!< aligned
        std::vector<float> m_TablesY;   //!< Lanczos table * m_NumCoordsY
        std::vector<int32_t> m_IndicesX;
        std::vector<float> m_Work;
    };

    template<>
    bool LanczosResizerImpl_hasFeature<ArchAVX512>()
    {
        HWCap cap;
        return cap.hasAVX512F()
            && cap.hasAVX512VL()
            && cap.hasAVX512BW()
            && cap.hasAVX512DQ()
            && cap.hasAVX512CD();
    }

    template<>
    ILanczosResizerImpl * LanczosResizerImpl_new<ArchAVX512>()
    {
        return new LanczosResizerImpl<ArchAVX512>();
    }


    // Constructor
    void LanczosResizerImpl<ArchAVX512>::init(
        unsigned int degree,
        size_t srcW, size_t srcH,
        size_t dstW, size_t dstH,
        size_t pxScale
    ) {
        m_SrcW = srcW;
        m_SrcH = srcH;
        m_DstW = dstW;
        m_DstH = dstH;

        // setup coefficients
        if ( m_SrcW <= m_DstW ) {
            // horizontal: up-sampling
            m_NumCoefsX = 2 * degree;
        } else {
            // vertical: down-sampling
            // n = 2*degree / scale
            size_t degree2 = std::max<size_t>(1, degree / pxScale);
            m_NumCoefsX = 2 * intptr_t(std::ceil((degree2 * m_SrcW) / double(m_DstW)));
        }
        if ( m_SrcH <= m_DstH ) {
            // vertical: up-sampling
            m_NumCoefsY = 2 * degree;
        } else {
            // vertical: down-sampling
            // n = 2*degree / scale
            size_t degree2 = std::max<size_t>(1, degree / pxScale);
            m_NumCoefsY = 2 * intptr_t(std::ceil((degree2 * m_SrcH) / double(m_DstH)));
        }
        m_NumCoordsX = m_DstW / gcd(m_SrcW, m_DstW);
        m_NumUnrolledCoordsX = std::min(alignCeil(m_DstW, kVecStepX), lcm(m_NumCoordsX, kVecStepX));
        m_TablesXWidth = kVecStepX * m_NumCoefsX;
        m_NumCoordsY = m_DstH / gcd(m_SrcH, m_DstH);
        m_TablesX_.reserve(m_TablesXWidth * m_NumCoordsX + kVecStepX);
        m_TablesX_.resize(m_TablesXWidth * m_NumCoordsX + kVecStepX);
        m_TablesX = (float *)alignCeil(intptr_t(&m_TablesX_[0]), sizeof(*m_TablesX) * kVecStepX);
        m_TablesY.reserve(m_NumCoefsY * m_NumCoordsY);
        m_TablesY.resize(m_NumCoefsY * m_NumCoordsY);

        std::vector<float> tablesX(m_NumCoefsX * m_NumCoordsX);
        for ( intptr_t dstX = 0; dstX < m_NumCoordsX; ++dstX ) {
            float * table = &tablesX[dstX * m_NumCoefsX];
            double sumCoefs = setLanczosTable(degree, m_SrcW, m_DstW, dstX, pxScale, m_NumCoefsX, table);
            for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
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
        intptr_t tblW = m_TablesXWidth;
        intptr_t nCoords = m_NumCoordsX;
        for ( intptr_t row = 0; row < nCoords; ++row ) {
            for ( intptr_t col = 0; col < tblW; ++col ) {
                intptr_t iCoef = col/kVecStepX;
                intptr_t srcX = (col%kVecStepX + row) % nCoords;
                intptr_t i = iCoef + srcX*m_NumCoefsX;
                m_TablesX[col + m_TablesXWidth*row] = tablesX[i];
            }
        }

        for ( intptr_t dstY = 0; dstY < m_NumCoordsY; ++dstY ) {
            float * table = &m_TablesY[dstY * m_NumCoefsY];
            double sumCoefs = setLanczosTable(degree, m_SrcH, m_DstH, dstY, pxScale, m_NumCoefsY, table);
            for ( intptr_t i = 0; i < m_NumCoefsY; ++i ) {
                table[i] /= sumCoefs;
            }
        }

        // allocate workspace
        m_Work.reserve(m_SrcW * getNumberOfProcs());
        m_Work.resize(m_SrcW * getNumberOfProcs());

        // calc indices
        intptr_t alignedDstW = alignCeil(m_DstW, kVecStepX);
        m_IndicesX.reserve(alignedDstW);
        m_IndicesX.resize(alignedDstW);
        for ( intptr_t dstX = 0; dstX < alignedDstW; ++dstX ) {
            //       srcOX = floor(dstX / scale)
            intptr_t srcOX = dstX * m_SrcW / m_DstW + 1;
            m_IndicesX[dstX] = srcOX;
        }
    }

    void LanczosResizerImpl<ArchAVX512>::resize(
        size_t srcSt, const uint8_t * src,
        size_t dstSt, uint8_t * __restrict dst
    ) {
        if ( m_SrcH == m_DstH ) {
#pragma omp parallel for
            for ( intptr_t y = 0; y < m_SrcH; ++y ) {
                float * work = &m_Work[getThreadNumber() * m_SrcW];
                for ( intptr_t x = 0; x < m_SrcW; ++x ) {
                    work[x] = src[srcSt * y + x];
                }
                resizeX(work, &dst[dstSt * y]);
            }
            return;
        }

        intptr_t numCoefsOn2 = m_NumCoefsY / 2;
        // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstH / double(m_SrcH))
        intptr_t mainBegin = ((numCoefsOn2 - 1) * m_DstH + m_SrcH-1) / m_SrcH;
        intptr_t mainEnd = std::max<intptr_t>(0, (m_SrcH - numCoefsOn2) * m_DstH / m_SrcH);
        const float * tablesY = &m_TablesY[0];

        // border pixels
#pragma omp parallel for
        for ( intptr_t dstY = 0; dstY < mainBegin; ++dstY ) {
            float * work = &m_Work[getThreadNumber() * m_SrcW];
            intptr_t srcOY = dstY * m_SrcH / m_DstH + 1;
            const float * coefs = &tablesY[dstY % m_NumCoordsY * m_NumCoefsY];
            resizeYborder(
                srcSt, &src[0],
                m_SrcW, work,
                srcOY,
                coefs);
            resizeX(work, &dst[dstSt * dstY]);
        }

        // main loop
#pragma omp parallel for
        for ( intptr_t dstY = mainBegin; dstY < mainEnd; ++dstY ) {
            float * work = &m_Work[getThreadNumber() * m_SrcW];
            intptr_t srcOY = dstY * m_SrcH / m_DstH + 1;
            const float * coefs = &tablesY[dstY % m_NumCoordsY * m_NumCoefsY];
            resizeYmain(
                srcSt, &src[0],
                m_SrcW, work,
                srcOY,
                coefs);
            resizeX(work, &dst[dstSt * dstY]);
        }

        // border pixels
#pragma omp parallel for
        for ( intptr_t dstY = mainEnd; dstY < m_DstH; ++dstY ) {
            float * work = &m_Work[getThreadNumber() * m_SrcW];
            intptr_t srcOY = dstY * m_SrcH / m_DstH + 1;
            const float * coefs = &tablesY[dstY % m_NumCoordsY * m_NumCoefsY];
            resizeYborder(
                srcSt, &src[0],
                m_SrcW, work,
                srcOY,
                coefs);
            resizeX(work, &dst[dstSt * dstY]);
        }
    }

    //! resize vertical (border loop)
    //!
    //! @param srcSt  Stride in src (in byte)
    //! @param src    A row of source
    //! @param dst    A row of destination
    //! @param srcOY  The origin of current line
    //! @param coefs  The coefficients
    void LanczosResizerImpl<ArchAVX512>::resizeYborder(
        intptr_t srcSt, const uint8_t * src,
        intptr_t dstW, float * __restrict dst,
        intptr_t srcOY,
        const float * coefs
    ) {
        intptr_t numCoefsOn2 = m_NumCoefsY / 2;
        intptr_t numCoefsY = m_NumCoefsY;
        intptr_t vecLen = alignFloor(dstW, kVecStepY);
        intptr_t srcH = m_SrcH;

        for ( intptr_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            // nume = 0;
            __m512 f32x16Nume0 = _mm512_setzero_ps();
            __m512 f32x16Nume1 = _mm512_setzero_ps();
            float deno = 0;

            for ( intptr_t i = 0; i < numCoefsY; ++i ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                if ( 0 <= srcY && srcY < srcH ) {
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
                    deno += coefs[i];
                }
            }

            // dst[dstX] = nume / deno;
            __m512 f32x16Deno = _mm512_set1_ps(deno);
            __m512 f32x16Dst0 = _mm512_div_ps(f32x16Nume0, f32x16Deno);
            __m512 f32x16Dst1 = _mm512_div_ps(f32x16Nume1, f32x16Deno);
            _mm512_storeu_ps(&dst[dstX +  0], f32x16Dst0);
            _mm512_storeu_ps(&dst[dstX + 16], f32x16Dst1);
        }

        for ( intptr_t dstX = vecLen; dstX < dstW; dstX++ ) {
            float nume = 0;
            float deno = 0;

            for ( intptr_t i = 0; i < numCoefsY; ++i ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                float    coef = coefs[i];
                nume += src[dstX + srcSt * srcY] * coef;
                deno += coef;
            }

            dst[dstX] = nume / deno;
        }
    }

    //! resize vertical (main loop)
    //!
    //! @param srcSt  Stride in src (in byte)
    //! @param src    A row of source
    //! @param dst    A row of destination
    //! @param srcOY  The origin of current line
    //! @param coefs  The coefficients
    void LanczosResizerImpl<ArchAVX512>::resizeYmain(
        intptr_t srcSt, const uint8_t * src,
        intptr_t dstW, float * __restrict dst,
        intptr_t srcOY,
        const float * coefs
    ) {
        intptr_t numCoefsOn2 = m_NumCoefsY / 2;
        intptr_t numCoefsY = m_NumCoefsY;
        intptr_t vecLen = alignFloor(dstW, kVecStepY);

        for ( intptr_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            // nume = 0;
            __m512 f32x16Nume0 = _mm512_setzero_ps();
            __m512 f32x16Nume1 = _mm512_setzero_ps();

            for ( intptr_t i = 0; i < numCoefsY; ++i ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
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

        for ( intptr_t dstX = vecLen; dstX < dstW; dstX++ ) {
            float nume = 0;

            for ( intptr_t i = 0; i < numCoefsY; ++i ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                float    coef = coefs[i];
                nume += src[dstX + srcSt * srcY] * coef;
            }

            dst[dstX] = nume;
        }
    }

    void LanczosResizerImpl<ArchAVX512>::resizeX(const float * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            intptr_t dstW = m_DstW;
            for ( intptr_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = cvt_roundss_su8(src[dstX]);
            }
            return;
        }

        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstW / double(m_SrcW))
        intptr_t mainBegin = alignCeil(((numCoefsOn2 - 1) * m_DstW + m_SrcW-1) / m_SrcW, kVecStepX);
        intptr_t mainEnd = std::max<intptr_t>(0, (m_SrcW - numCoefsOn2) * m_DstW / m_SrcW);
        intptr_t mainLen = alignFloor(mainEnd - mainBegin, kVecStepX);
        mainEnd = mainBegin + mainLen;

        resizeXborder(src, dst, 0, mainBegin);
        resizeXmain(src, dst, mainBegin, mainEnd);
        resizeXborder(src, dst, mainEnd, m_DstW);
    }

    //! resize horizontal (border loop)
    //!
    //! @param src    A row of source
    //! @param dst    A row of destination
    //! @param begin  Position of a first pixel
    //! @param end    Position of next of a last pixel
    void LanczosResizerImpl<ArchAVX512>::resizeXborder(
        const float * src, uint8_t * __restrict dst,
        intptr_t begin, intptr_t end
    ) {
        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        intptr_t numCoefsX = m_NumCoefsX;
        intptr_t tableSize = m_TablesXWidth * m_NumCoordsX;
        const float * coefs = &m_TablesX[0];
        const int32_t * indices = &m_IndicesX[0];
        intptr_t iCoef = begin / kVecStepX % m_NumCoordsX * m_TablesXWidth;

        const __m512i s32x16k_1  = _mm512_set1_epi32(-1);
        const __m512i s32x16SrcW = _mm512_set1_epi32(m_SrcW);

        for ( intptr_t dstX = begin; dstX < end; dstX += kVecStepX ) {
            //      nume   = 0;
            __m512  f32x16Nume0 = _mm512_setzero_ps();
            __m512  f32x16Nume1 = _mm512_setzero_ps();
            //      deno   = 0;
            __m512  f32x16Deno0 = _mm512_setzero_ps();
            __m512  f32x16Deno1 = _mm512_setzero_ps();
            //      srcX = floor(dstX / scale) - numCoefsOn2 + 1 + i;
            __m512i s32x16SrcOX0 = _mm512_loadu_si512((const __m512i*)&indices[dstX +  0]);
            __m512i s32x16SrcOX1 = _mm512_loadu_si512((const __m512i*)&indices[dstX + 16]);

            for ( intptr_t i = 0; i < numCoefsX; ++i ) {
                __m512i s32x16Offset = _mm512_set1_epi32(i - numCoefsOn2);

                //intptr_t srcX = indices[dstX] + offset;
                __m512i s32x16SrcX0 = _mm512_add_epi32(s32x16SrcOX0, s32x16Offset);
                __m512i s32x16SrcX1 = _mm512_add_epi32(s32x16SrcOX1, s32x16Offset);

                // if ( 0 <= srcX && srcX < m_SrcW )
                __mmask16 b16Mask0 = _mm512_cmpgt_epi32_mask(s32x16SrcX0, s32x16k_1) & _mm512_cmpgt_epi32_mask(s32x16SrcW, s32x16SrcX0);
                __mmask16 b16Mask1 = _mm512_cmpgt_epi32_mask(s32x16SrcX1, s32x16k_1) & _mm512_cmpgt_epi32_mask(s32x16SrcW, s32x16SrcX1);

                // nume[dstX] += src[srcX] * coefs[iCoef];
                __m512  f32x16Src0    = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), b16Mask0, s32x16SrcX0, src, sizeof(float));
                __m512  f32x16Src1    = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), b16Mask1, s32x16SrcX1, src, sizeof(float));
                __m512  f32x16Coefs0  = _mm512_load_ps(&coefs[iCoef +  0]);
                __m512  f32x16Coefs1  = _mm512_load_ps(&coefs[iCoef + 16]);
                f32x16Nume0 = _mm512_fmadd_ps(f32x16Src0, f32x16Coefs0, f32x16Nume0);
                f32x16Nume1 = _mm512_fmadd_ps(f32x16Src1, f32x16Coefs1, f32x16Nume1);
                f32x16Deno0 = _mm512_add_ps(f32x16Deno0, f32x16Coefs0);
                f32x16Deno1 = _mm512_add_ps(f32x16Deno1, f32x16Coefs1);

                iCoef += kVecStepX;
            }

            // dst[dstX] = clamp<int>(0, 255, round(f32Nume / f32Deno));
            __m512  f32x16Dst0 = _mm512_div_ps(f32x16Nume0, f32x16Deno0);
            __m512  f32x16Dst1 = _mm512_div_ps(f32x16Nume1, f32x16Deno1);
            __m256i u8x32Dst = cvt_roundps_epu8(f32x16Dst0, f32x16Dst1);
            _mm256_storeu_si256((__m256i*)&dst[dstX], u8x32Dst);

            // iCoef = dstX % tableSize;
            if ( iCoef == tableSize ) {
                iCoef = 0;
            }
        }
    }

    //! resize horizontal (main loop)
    //!
    //! @param src    A row of source
    //! @param dst    A row of destination
    //! @param begin  Position of a first pixel
    //! @param end    Position of next of a last pixel
    void LanczosResizerImpl<ArchAVX512>::resizeXmain(
        const float * src, uint8_t * __restrict dst,
        intptr_t begin, intptr_t end
    ) {
        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        intptr_t numCoefsX = m_NumCoefsX;
        intptr_t tableSize = m_TablesXWidth * m_NumCoordsX;
        const float * coefs = &m_TablesX[0];
        const int32_t * indices = &m_IndicesX[0];
        intptr_t iCoef = begin / kVecStepX % m_NumCoordsX * m_TablesXWidth;

        for ( intptr_t dstX = begin; dstX < end; dstX += kVecStepX ) {
            //      nume   = 0;
            __m512  f32x16Nume0 = _mm512_setzero_ps();
            __m512  f32x16Nume1 = _mm512_setzero_ps();
            //      srcX = floor(dstX / scale) - numCoefsOn2 + 1 + i;
            __m512i s32x16SrcOX0 = _mm512_loadu_si512((const __m512i*)&indices[dstX +  0]);
            __m512i s32x16SrcOX1 = _mm512_loadu_si512((const __m512i*)&indices[dstX + 16]);

            for ( intptr_t i = 0; i < numCoefsX; ++i ) {
                __m512i s32x16Offset   = _mm512_set1_epi32(i - numCoefsOn2);

                //intptr_t srcX = indices[dstX] + offset;
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
    }

}

#else

namespace iqo {

    template<>
    bool LanczosResizerImpl_hasFeature<ArchAVX512>()
    {
        return false;
    }

    template<>
    ILanczosResizerImpl * LanczosResizerImpl_new<ArchAVX512>()
    {
        return NULL;
    }

}

#endif