#include "IQOLanczosResizerImpl.hpp"


#if defined(IQO_CPU_X86) && defined(IQO_HAVE_AVX2FMA)

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

    //! (uint8_t)min(255, max(0, v))
    __m128i packus_epi16(__m256i v)
    {
        __m128i s16x8L = _mm256_castsi256_si128(v);
        __m128i s16x8H = _mm256_extractf128_si256(v, 1);
        return _mm_packus_epi16(s16x8L, s16x8H);
    }

    __m256i packus_epi32(__m256i lo, __m256i hi)
    {
        __m256i u16x16Perm = _mm256_packus_epi32(lo, hi);
        return _mm256_permute4x64_epi64(u16x16Perm, _MM_SHUFFLE(3, 1, 2, 0));
    }

    uint8_t cvt_roundss_su8(float v)
    {
        __m128  f32x1V     = _mm_set_ss(v);
        __m128  f32x1Round = _mm_round_ss(f32x1V, f32x1V, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        int32_t s32x1Round = _mm_cvtss_si32(f32x1Round);
        return uint8_t(clamp(0, 255, s32x1Round));
    }

    //! (uint8_t)min(255, max(0, round(v)))
    __m128i cvt_roundps_epu8(__m256 lo, __m256 hi)
    {
        __m256  f32x8L = _mm256_round_ps(lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256  f32x8H = _mm256_round_ps(hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i s32x8L = _mm256_cvttps_epi32(f32x8L);
        __m256i s32x8H = _mm256_cvttps_epi32(f32x8H);
        __m256i u16x16 = packus_epi32(s32x8L, s32x8H);
        return packus_epi16(u16x16);
    }

}

namespace iqo {

    template<>
    class LanczosResizerImpl<ArchAVX2FMA> : public ILanczosResizerImpl
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
            ptrdiff_t srcSt, const uint8_t * src,
            ptrdiff_t dstW, float * dst,
            ptrdiff_t srcOY,
            const float * coefs);
        void resizeYmain(
            ptrdiff_t srcSt, const uint8_t * src,
            ptrdiff_t dstW, float * dst,
            ptrdiff_t srcOY,
            const float * coefs);
        void resizeX(const float * src, uint8_t * dst);
        void resizeXborder(
            const float * src, uint8_t * dst,
            ptrdiff_t begin, ptrdiff_t end);
        void resizeXmain(
            const float * src, uint8_t * dst,
            ptrdiff_t begin, ptrdiff_t end);

        enum {
            //! for SIMD
            kVecStepX   = 16, //!< __m256 x 2
            kVecStepY   = 16, //!< __m256 x 2
        };
        int32_t   m_SrcW;
        ptrdiff_t m_SrcH;
        ptrdiff_t m_DstW;
        ptrdiff_t m_DstH;
        int32_t   m_NumCoefsX;
        ptrdiff_t m_NumCoefsY;
        ptrdiff_t m_NumCoordsX;
        ptrdiff_t m_NumUnrolledCoordsX;
        ptrdiff_t m_TablesXWidth;
        ptrdiff_t m_NumCoordsY;
        std::vector<float> m_TablesX_;  //!< Lanczos table * m_NumCoordsX (unrolled)
        float * m_TablesX;              //!< aligned
        std::vector<float> m_TablesY;   //!< Lanczos table * m_NumCoordsY
        std::vector<int32_t> m_IndicesX;
        std::vector<float> m_Work;
    };

    template<>
    bool LanczosResizerImpl_hasFeature<ArchAVX2FMA>()
    {
        HWCap cap;
        return cap.hasAVX2() && cap.hasFMA();
    }

    template<>
    ILanczosResizerImpl * LanczosResizerImpl_new<ArchAVX2FMA>()
    {
        return new LanczosResizerImpl<ArchAVX2FMA>();
    }


    // Constructor
    void LanczosResizerImpl<ArchAVX2FMA>::init(
        unsigned int degree,
        size_t srcW, size_t srcH,
        size_t dstW, size_t dstH,
        size_t pxScale
    ) {
        m_SrcW = int32_t(srcW);
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
        m_NumCoefsX = int32_t(calcNumCoefsForLanczos(degree, rSrcW, rDstW, pxScale));
        m_NumCoefsY = calcNumCoefsForLanczos(degree, rSrcH, rDstH, pxScale);
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
            float sumCoefs = setLanczosTable(degree, rSrcW, rDstW, dstX, pxScale, m_NumCoefsX, table);
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
            float sumCoefs = setLanczosTable(degree, rSrcH, rDstH, dstY, pxScale, m_NumCoefsY, table);
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
            int32_t srcOX = int32_t(dstX * rSrcW / rDstW + 1);
            m_IndicesX[dstX] = srcOX;
        }
    }

    void LanczosResizerImpl<ArchAVX2FMA>::resize(
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

        // vertical
        ptrdiff_t numCoefsOn2 = m_NumCoefsY / 2;
        // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstH / double(m_SrcH))
        ptrdiff_t mainBegin = ((numCoefsOn2 - 1) * m_DstH + m_SrcH-1) / m_SrcH;
        ptrdiff_t mainEnd = std::max<ptrdiff_t>(0, (m_SrcH - numCoefsOn2) * m_DstH / m_SrcH);
        const float * tablesY = &m_TablesY[0];

        // border pixels
#pragma omp parallel for
        for ( ptrdiff_t dstY = 0; dstY < mainBegin; ++dstY ) {
            float * work = &m_Work[getThreadNumber() * m_SrcW];
            ptrdiff_t srcOY = dstY * m_SrcH / m_DstH + 1;
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
        for ( ptrdiff_t dstY = mainBegin; dstY < mainEnd; ++dstY ) {
            float * work = &m_Work[getThreadNumber() * m_SrcW];
            ptrdiff_t srcOY = dstY * m_SrcH / m_DstH + 1;
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
        for ( ptrdiff_t dstY = mainEnd; dstY < m_DstH; ++dstY ) {
            float * work = &m_Work[getThreadNumber() * m_SrcW];
            ptrdiff_t srcOY = dstY * m_SrcH / m_DstH + 1;
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
    void LanczosResizerImpl<ArchAVX2FMA>::resizeYborder(
        ptrdiff_t srcSt, const uint8_t * src,
        ptrdiff_t dstW, float * __restrict dst,
        ptrdiff_t srcOY,
        const float * coefs
    ) {
        ptrdiff_t numCoefsOn2 = m_NumCoefsY / 2;
        ptrdiff_t numCoefsY = m_NumCoefsY;
        ptrdiff_t vecLen = alignFloor(dstW, kVecStepY);
        ptrdiff_t srcH = m_SrcH;

        for ( ptrdiff_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            // nume = 0;
            __m256 f32x8Nume0 = _mm256_setzero_ps();
            __m256 f32x8Nume1 = _mm256_setzero_ps();
            float deno = 0;

            for ( ptrdiff_t i = 0; i < numCoefsY; ++i ) {
                ptrdiff_t srcY = srcOY - numCoefsOn2 + i;
                if ( 0 <= srcY && srcY < srcH ) {
                    // coef = coefs[i];
                    __m256  f32x8Coef = _mm256_set1_ps(coefs[i]);
                    // nume += src[dstX + srcSt*srcY] * coef;
                    __m128i u8x16Src    = _mm_loadu_si128((const __m128i *)&src[dstX + srcSt*srcY]);
                    __m128i u8x8Src0    = u8x16Src;
                    __m128i u8x8Src1    = _mm_shuffle_epi32(u8x16Src, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256  f32x8Src0   = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u8x8Src0));
                    __m256  f32x8Src1   = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u8x8Src1));
                    f32x8Nume0 = _mm256_fmadd_ps(f32x8Src0, f32x8Coef, f32x8Nume0);
                    f32x8Nume1 = _mm256_fmadd_ps(f32x8Src1, f32x8Coef, f32x8Nume1);
                    deno += coefs[i];
                }
            }

            // dst[dstX] = nume / deno;
            __m256 f32x8Deno = _mm256_set1_ps(deno);
            __m256 f32x8Dst0 = _mm256_div_ps(f32x8Nume0, f32x8Deno);
            __m256 f32x8Dst1 = _mm256_div_ps(f32x8Nume1, f32x8Deno);
            _mm256_storeu_ps(&dst[dstX + 0], f32x8Dst0);
            _mm256_storeu_ps(&dst[dstX + 8], f32x8Dst1);
        }

        for ( ptrdiff_t dstX = vecLen; dstX < dstW; dstX++ ) {
            float nume = 0;
            float deno = 0;

            for ( ptrdiff_t i = 0; i < numCoefsY; ++i ) {
                ptrdiff_t srcY = srcOY - numCoefsOn2 + i;
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
    void LanczosResizerImpl<ArchAVX2FMA>::resizeYmain(
        ptrdiff_t srcSt, const uint8_t * src,
        ptrdiff_t dstW, float * __restrict dst,
        ptrdiff_t srcOY,
        const float * coefs
    ) {
        ptrdiff_t numCoefsOn2 = m_NumCoefsY / 2;
        ptrdiff_t numCoefsY = m_NumCoefsY;
        ptrdiff_t vecLen = alignFloor(dstW, kVecStepY);

        for ( ptrdiff_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            // nume = 0;
            __m256 f32x8Nume0 = _mm256_setzero_ps();
            __m256 f32x8Nume1 = _mm256_setzero_ps();

            for ( ptrdiff_t i = 0; i < numCoefsY; ++i ) {
                ptrdiff_t srcY = srcOY - numCoefsOn2 + i;
                // coef = coefs[i];
                __m256  f32x8Coef = _mm256_set1_ps(coefs[i]);
                // nume += src[dstX + srcSt*srcY] * coef;
                __m128i u8x16Src    = _mm_loadu_si128((const __m128i *)&src[dstX + srcSt*srcY]);
                __m128i u8x8Src0    = u8x16Src;
                __m128i u8x8Src1    = _mm_shuffle_epi32(u8x16Src, _MM_SHUFFLE(3, 2, 3, 2));
                __m256  f32x8Src0   = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u8x8Src0));
                __m256  f32x8Src1   = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u8x8Src1));
                f32x8Nume0 = _mm256_fmadd_ps(f32x8Src0, f32x8Coef, f32x8Nume0);
                f32x8Nume1 = _mm256_fmadd_ps(f32x8Src1, f32x8Coef, f32x8Nume1);
            }

            // dst[dstX] = nume;
            __m256 f32x8Dst0 = f32x8Nume0;
            __m256 f32x8Dst1 = f32x8Nume1;
            _mm256_storeu_ps(&dst[dstX + 0], f32x8Dst0);
            _mm256_storeu_ps(&dst[dstX + 8], f32x8Dst1);
        }

        for ( ptrdiff_t dstX = vecLen; dstX < dstW; dstX++ ) {
            float nume = 0;

            for ( ptrdiff_t i = 0; i < numCoefsY; ++i ) {
                ptrdiff_t srcY = srcOY - numCoefsOn2 + i;
                float    coef = coefs[i];
                nume += src[dstX + srcSt * srcY] * coef;
            }

            dst[dstX] = nume;
        }
    }

    void LanczosResizerImpl<ArchAVX2FMA>::resizeX(const float * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            ptrdiff_t dstW = m_DstW;
            for ( ptrdiff_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = cvt_roundss_su8(src[dstX]);
            }
            return;
        }

        ptrdiff_t numCoefsOn2 = m_NumCoefsX / 2;
        // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstW / double(m_SrcW))
        ptrdiff_t mainBegin = alignCeil(((numCoefsOn2 - 1) * m_DstW + m_SrcW-1) / m_SrcW, kVecStepX);
        ptrdiff_t mainEnd = std::max<ptrdiff_t>(0, (m_SrcW - numCoefsOn2) * m_DstW / m_SrcW);
        ptrdiff_t mainLen = alignFloor(mainEnd - mainBegin, kVecStepX);
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
    void LanczosResizerImpl<ArchAVX2FMA>::resizeXborder(
        const float * src, uint8_t * __restrict dst,
        ptrdiff_t begin, ptrdiff_t end
    ) {
        int32_t numCoefsOn2 = m_NumCoefsX / 2;
        const float * coefs = &m_TablesX[0];
        const int32_t * indices = &m_IndicesX[0];
        ptrdiff_t tableSize = m_TablesXWidth * m_NumUnrolledCoordsX;
        int32_t numCoefsX = m_NumCoefsX;

        __m256i s32x8k_1  = _mm256_set1_epi32(-1);
        __m256i s32x8SrcW = _mm256_set1_epi32(m_SrcW);
        ptrdiff_t iCoef = begin / kVecStepX % m_NumUnrolledCoordsX * m_TablesXWidth;
        for ( ptrdiff_t dstX = begin; dstX < end; dstX += kVecStepX ) {
            //      nume   = 0;
            __m256  f32x8Nume0 = _mm256_setzero_ps();
            __m256  f32x8Nume8 = _mm256_setzero_ps();
            //      deno   = 0;
            __m256  f32x8Deno0 = _mm256_setzero_ps();
            __m256  f32x8Deno8 = _mm256_setzero_ps();
            //      srcX = floor(dstX / scale) - numCoefsOn2 + 1 + i;
            __m256i s32x8SrcOX0 = _mm256_loadu_si256((const __m256i*)&indices[dstX + 0]);
            __m256i s32x8SrcOX8 = _mm256_loadu_si256((const __m256i*)&indices[dstX + 8]);

            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                __m256i s32x8Offset = _mm256_set1_epi32(i - numCoefsOn2);

                //ptrdiff_t srcX = indices[dstX + j] + offset;
                __m256i s32x8SrcX0 = _mm256_add_epi32(s32x8SrcOX0, s32x8Offset);
                __m256i s32x8SrcX8 = _mm256_add_epi32(s32x8SrcOX8, s32x8Offset);

                // if ( 0 <= srcX && srcX < m_SrcW )
                __m256i s32x8Mask0   = _mm256_and_si256(_mm256_cmpgt_epi32(s32x8SrcX0, s32x8k_1), _mm256_cmpgt_epi32(s32x8SrcW, s32x8SrcX0));
                __m256i s32x8Mask8   = _mm256_and_si256(_mm256_cmpgt_epi32(s32x8SrcX8, s32x8k_1), _mm256_cmpgt_epi32(s32x8SrcW, s32x8SrcX8));
                __m256  f32x8Mask0   = _mm256_castsi256_ps(s32x8Mask0);
                __m256  f32x8Mask8   = _mm256_castsi256_ps(s32x8Mask8);

                //nume[dstX + j] += src[srcX] * coefs[dstX + j];
                __m256  f32x8Src0    = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src, s32x8SrcX0, f32x8Mask0, sizeof(float));
                __m256  f32x8Src8    = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src, s32x8SrcX8, f32x8Mask8, sizeof(float));
                __m256  f32x8Coefs0  = _mm256_load_ps(&coefs[iCoef + 0]);
                __m256  f32x8Coefs8  = _mm256_load_ps(&coefs[iCoef + 8]);
                f32x8Nume0 = _mm256_fmadd_ps(f32x8Src0, f32x8Coefs0, f32x8Nume0);
                f32x8Nume8 = _mm256_fmadd_ps(f32x8Src8, f32x8Coefs8, f32x8Nume8);
                f32x8Deno0 = _mm256_add_ps(f32x8Deno0, f32x8Coefs0);
                f32x8Deno8 = _mm256_add_ps(f32x8Deno8, f32x8Coefs8);

                iCoef += kVecStepX;
            }

            // dst[dstX] = clamp<int>(0, 255, round(f32Nume / f32Deno));
            __m256  f32x8Dst0 = _mm256_div_ps(f32x8Nume0, f32x8Deno0);
            __m256  f32x8Dst8 = _mm256_div_ps(f32x8Nume8, f32x8Deno8);
            __m128i u8x16Dst = cvt_roundps_epu8(f32x8Dst0, f32x8Dst8);
            _mm_storeu_si128((__m128i*)&dst[dstX], u8x16Dst);

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
    void LanczosResizerImpl<ArchAVX2FMA>::resizeXmain(
        const float * src, uint8_t * __restrict dst,
        ptrdiff_t begin, ptrdiff_t end
    ) {
        int32_t numCoefsOn2 = m_NumCoefsX / 2;
        const float * coefs = &m_TablesX[0];
        const int32_t * indices = &m_IndicesX[0];
        ptrdiff_t tableSize = m_TablesXWidth * m_NumUnrolledCoordsX;
        int32_t numCoefsX = m_NumCoefsX;

        ptrdiff_t iCoef = begin / kVecStepX % m_NumUnrolledCoordsX * m_TablesXWidth;
        for ( ptrdiff_t dstX = begin; dstX < end; dstX += kVecStepX ) {
            //      nume   = 0;
            __m256  f32x8Nume0 = _mm256_setzero_ps();
            __m256  f32x8Nume8 = _mm256_setzero_ps();
            //       srcX = floor(dstX / scale) - numCoefsOn2 + 1 + i;
            __m256i s32x8SrcOX0 = _mm256_loadu_si256((const __m256i*)&indices[dstX + 0]);
            __m256i s32x8SrcOX8 = _mm256_loadu_si256((const __m256i*)&indices[dstX + 8]);

            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                __m256i s32x8Offset   = _mm256_set1_epi32(i - numCoefsOn2);

                //ptrdiff_t srcX = indices[dstX + j] + offset;
                __m256i s32x8SrcX0 = _mm256_add_epi32(s32x8SrcOX0, s32x8Offset);
                __m256i s32x8SrcX8 = _mm256_add_epi32(s32x8SrcOX8, s32x8Offset);

                //nume[dstX + j] += src[srcX] * coefs[dstX + j];
                __m256  s32x8Src0    = _mm256_i32gather_ps(src, s32x8SrcX0, sizeof(float));
                __m256  s32x8Src8    = _mm256_i32gather_ps(src, s32x8SrcX8, sizeof(float));
                __m256  f32x8Coefs0  = _mm256_load_ps(&coefs[iCoef + 0]);
                __m256  f32x8Coefs8  = _mm256_load_ps(&coefs[iCoef + 8]);
                f32x8Nume0 = _mm256_fmadd_ps(s32x8Src0, f32x8Coefs0, f32x8Nume0);
                f32x8Nume8 = _mm256_fmadd_ps(s32x8Src8, f32x8Coefs8, f32x8Nume8);
                iCoef += kVecStepX;
            }

            __m128i u8x16Dst = cvt_roundps_epu8(f32x8Nume0, f32x8Nume8);
            _mm_storeu_si128((__m128i*)&dst[dstX], u8x16Dst);

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
    bool LanczosResizerImpl_hasFeature<ArchAVX2FMA>()
    {
        return false;
    }

    template<>
    ILanczosResizerImpl * LanczosResizerImpl_new<ArchAVX2FMA>()
    {
        return NULL;
    }

}

#endif
