#include "IQOLinearResizerImpl.hpp"


#if defined(IQO_CPU_X86) && defined(IQO_HAVE_AVX2FMA)

#include <cstring>
#include <vector>
#include <immintrin.h>

#if defined(_OPENMP)
    #include <omp.h>
#endif

#include "math.hpp"
#include "IQOHWCap.hpp"


namespace {

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
        return uint8_t(iqo::clamp(0, 255, s32x1Round));
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
    class LinearResizerImpl<ArchAVX2FMA> : public ILinearResizerImpl
    {
    public:
        //! Destructor
        virtual ~LinearResizerImpl() {}

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
        void resizeYborder(
            ptrdiff_t srcSt, const uint8_t * src,
            int32_t dstW, float * dst,
            int32_t srcOY
        );
        void resizeYmain(
            ptrdiff_t srcSt, const uint8_t * src,
            int32_t dstW, float * dst,
            int32_t srcOY,
            const float * coefs
        );
        void resizeX(const float * src, uint8_t * dst);
        void resizeXborder(
            const float * src, uint8_t * dst,
            int32_t mainBegin, int32_t mainEnd,
            int32_t begin, int32_t end
        );
        void resizeXmain(
            const float * src, uint8_t * dst,
            int32_t begin, int32_t end
        );

        enum {
            m_NumCoefsX = 2,
            m_NumCoefsY = 2,

            //! for SIMD
            kVecStepX   = 16, //!< __m256 x 2
            kVecStepY   = 16, //!< __m256 x 2
        };
        int32_t m_SrcW;
        int32_t m_SrcH;
        int32_t m_DstW;
        int32_t m_DstH;
        int32_t m_NumCoordsX;
        int32_t m_NumUnrolledCoordsX;
        int32_t m_TablesXWidth;
        int32_t m_NumCoordsY;
        std::vector<float> m_TablesX_;  //!< Linear table * m_NumCoordsX (unrolled)
        float * m_TablesX;              //!< aligned
        std::vector<float> m_TablesY;   //!< Linear table * m_NumCoordsY
        std::vector<int32_t> m_IndicesX;
        std::vector<float> m_Work;
    };

    template<>
    bool LinearResizerImpl_hasFeature<ArchAVX2FMA>()
    {
        HWCap cap;
        return cap.hasAVX2() && cap.hasFMA();
    }

    template<>
    ILinearResizerImpl * LinearResizerImpl_new<ArchAVX2FMA>()
    {
        return new LinearResizerImpl<ArchAVX2FMA>();
    }


    // Constructor
    void LinearResizerImpl<ArchAVX2FMA>::init(
        size_t srcW, size_t srcH,
        size_t dstW, size_t dstH
    ) {
        m_SrcW = int32_t(srcW);
        m_SrcH = int32_t(srcH);
        m_DstW = int32_t(dstW);
        m_DstH = int32_t(dstH);

        // setup coefficients
        int32_t gcdW = int32_t(gcd(m_SrcW, m_DstW));
        int32_t gcdH = int32_t(gcd(m_SrcH, m_DstH));
        int32_t rSrcW = m_SrcW / gcdW;
        int32_t rDstW = m_DstW / gcdW;
        int32_t rSrcH = m_SrcH / gcdH;
        int32_t rDstH = m_DstH / gcdH;
        int64_t alignedW  = alignCeil<int32_t>(m_DstW, kVecStepX);
        int64_t unrolledW = lcm(rDstW, kVecStepX);
        m_NumCoordsX = rDstW;
        m_NumUnrolledCoordsX = int32_t(std::min(alignedW, unrolledW));
        m_NumUnrolledCoordsX /= kVecStepX;
        m_TablesXWidth = kVecStepX * m_NumCoefsX;
        m_NumCoordsY = rDstH;
        size_t alignedTblXSize = size_t(m_TablesXWidth) * m_NumUnrolledCoordsX + kVecStepX;
        m_TablesX_.reserve(alignedTblXSize);
        m_TablesX_.resize(alignedTblXSize);
        intptr_t addrTablesX_ = intptr_t(&m_TablesX_[0]);
        intptr_t addrTablesX  = alignCeil<intptr_t>(addrTablesX_, sizeof(*m_TablesX) * kVecStepX);
        m_TablesX = reinterpret_cast<float *>(addrTablesX);
        size_t tblYSize = size_t(m_NumCoefsY) * m_NumCoordsY;
        m_TablesY.reserve(tblYSize);
        m_TablesY.resize(tblYSize);

        // X coefs
        std::vector<float> tablesX(m_NumCoefsX * m_NumCoordsX);
        setLinearTable(rSrcW, rDstW, &tablesX[0]);

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
        setLinearTable(rSrcH, rDstH, &m_TablesY[0]);

        // allocate workspace
        m_Work.reserve(m_SrcW * HWCap::getNumberOfProcs());
        m_Work.resize(m_SrcW * HWCap::getNumberOfProcs());

        // calc indices
        int32_t alignedDstW = alignCeil<int32_t>(m_DstW, kVecStepX);
        m_IndicesX.reserve(alignedDstW);
        m_IndicesX.resize(alignedDstW);
        LinearIterator iSrcOX(dstW, srcW);
        iSrcOX.setX(srcW - dstW, 2 * dstW); // align center
         for ( int32_t dstX = 0; dstX < alignedDstW; ++dstX ) {
            //int32_t srcOX = floor((dstX+0.5) * srcW/dstW - 0.5);
            int32_t srcOX = int32_t(*iSrcOX++);
            m_IndicesX[dstX] = srcOX;
        }
    }

    void LinearResizerImpl<ArchAVX2FMA>::resize(
        size_t srcSt, const uint8_t * src,
        size_t dstSt, uint8_t * __restrict dst
    ) {
        int32_t srcW = m_SrcW;
        int32_t srcH = m_SrcH;
        int32_t dstH = m_DstH;

        if ( srcH == dstH ) {
#pragma omp parallel for
            for ( int32_t y = 0; y < srcH; ++y ) {
                float * work = &m_Work[HWCap::getThreadNumber() * ptrdiff_t(srcW)];
                for ( int32_t x = 0; x < srcW; ++x ) {
                    work[x] = src[srcSt * y + x];
                }
                resizeX(work, &dst[dstSt * y]);
            }
            return;
        }

        const float * tablesY = &m_TablesY[0];
        double        fMainBegin = std::ceil(0.5 * dstH / srcH - 0.5);
        ptrdiff_t     mainBegin  = clamp<ptrdiff_t>(0, dstH, ptrdiff_t(fMainBegin));
        ptrdiff_t     mainEnd    = clamp<ptrdiff_t>(0, dstH, dstH - mainBegin);

        // border pixels
#pragma omp parallel for
        for ( ptrdiff_t dstY = 0; dstY < mainBegin; ++dstY ) {
            float * work = &m_Work[HWCap::getThreadNumber() * ptrdiff_t(srcW)];
            int32_t srcOY = 0;
            resizeYborder(
                srcSt, &src[0],
                srcW, work,
                srcOY);
            resizeX(work, &dst[dstSt * dstY]);
        }

        // main loop
#pragma omp parallel for
        for ( ptrdiff_t dstY = mainBegin; dstY < mainEnd; ++dstY ) {
            float * work = &m_Work[HWCap::getThreadNumber() * ptrdiff_t(srcW)];
            int32_t srcOY = int32_t(floor((dstY+0.5) * srcH/dstH - 0.5));
            const float * coefs = &tablesY[dstY % m_NumCoordsY * ptrdiff_t(m_NumCoefsY)];
            resizeYmain(
                srcSt, &src[0],
                srcW, work,
                srcOY,
                coefs);
            resizeX(work, &dst[dstSt * dstY]);
        }

        // border pixels
#pragma omp parallel for
        for ( ptrdiff_t dstY = mainEnd; dstY < m_DstH; ++dstY ) {
            float * work = &m_Work[HWCap::getThreadNumber() * ptrdiff_t(srcW)];
            int32_t srcOY = srcH - 1;
            resizeYborder(
                srcSt, &src[0],
                srcW, work,
                srcOY);
            resizeX(work, &dst[dstSt * dstY]);
        }
    }

    //! resize vertical (border loop)
    //!
    //! @param srcSt  Stride in src (in byte)
    //! @param src    A row of source
    //! @param dst    A row of destination (multiplied by kBias)
    //! @param srcOY  The origin of current line
    void LinearResizerImpl<ArchAVX2FMA>::resizeYborder(
        ptrdiff_t srcSt, const uint8_t * src,
        int32_t dstW, float * __restrict dst,
        int32_t srcOY
    ) {
        for ( ptrdiff_t dstX = 0; dstX < dstW; ++dstX ) {
            dst[dstX] = src[dstX + srcSt * srcOY];
        }
    }

    //! resize vertical (main loop)
    //!
    //! @param srcSt  Stride in src (in byte)
    //! @param src    A row of source
    //! @param dst    A row of destination
    //! @param srcOY  The origin of current line
    //! @param coefs  The coefficients
    void LinearResizerImpl<ArchAVX2FMA>::resizeYmain(
        ptrdiff_t srcSt, const uint8_t * src,
        int32_t dstW, float * __restrict dst,
        int32_t srcOY,
        const float * coefs
    ) {
        int32_t numCoefsY = m_NumCoefsY;
        int32_t vecLen = alignFloor<int32_t>(dstW, kVecStepY);

        for ( int32_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            //     nume         = 0;
            __m256 f32x8Nume0   = _mm256_setzero_ps();
            __m256 f32x8Nume1   = _mm256_setzero_ps();

            for ( int32_t i = 0; i < numCoefsY; ++i ) {
                int32_t srcY = srcOY + i;

                //      coef        = coefs[i];
                __m256  f32x8Coef   = _mm256_set1_ps(coefs[i]);

                //      nume       += src[dstX + srcSt*srcY] * coef;
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

        for ( int32_t dstX = vecLen; dstX < dstW; dstX++ ) {
            float nume = 0;

            for ( int32_t i = 0; i < numCoefsY; ++i ) {
                int32_t srcY = srcOY + i;
                float   coef = coefs[i];
                nume += src[dstX + srcSt * srcY] * coef;
            }

            dst[dstX] = nume;
        }
    }

    void LinearResizerImpl<ArchAVX2FMA>::resizeX(const float * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            int32_t dstW = m_DstW;
            for ( int32_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = cvt_roundss_su8(src[dstX]);
            }
            return;
        }

        int32_t dstW        = m_DstW;
        double  fMainBegin  = std::ceil(0.5 * dstW / m_SrcW - 0.5);
        int32_t mainBegin   = clamp<int32_t>(0, dstW, int32_t(fMainBegin));
        int32_t mainEnd     = clamp<int32_t>(0, dstW, dstW - mainBegin);
        int32_t vecBegin    = alignCeil<int32_t>(mainBegin, kVecStepX);
        int32_t vecLen      = alignFloor<int32_t>(mainEnd - vecBegin, kVecStepX);
        int32_t vecEnd      = vecBegin + vecLen;

        resizeXborder(src, dst, mainBegin, mainEnd, 0, vecBegin);
        resizeXmain(src, dst, vecBegin, vecEnd);
        resizeXborder(src, dst, mainBegin, mainEnd, vecEnd, dstW);
    }

    //! resize horizontal (border loop)
    //!
    //! @param src    A row of source
    //! @param dst    A row of destination
    //! @param begin  Position of a first pixel
    //! @param end    Position of next of a last pixel
    void LinearResizerImpl<ArchAVX2FMA>::resizeXborder(
        const float * src, uint8_t * __restrict dst,
        int32_t mainBegin, int32_t mainEnd,
        int32_t begin, int32_t end
    ) {
        const float * coefs = &m_TablesX[0];
        const int32_t * indices = &m_IndicesX[0];
        ptrdiff_t tableWidth = ptrdiff_t(m_TablesXWidth);
        ptrdiff_t tableSize = tableWidth * m_NumUnrolledCoordsX;
        int32_t numCoefsX = m_NumCoefsX;
        int32_t srcW = m_SrcW;
        int32_t dstW = m_DstW;
        LinearIterator iSrcOX(dstW, srcW);

        // align center
        iSrcOX.setX(srcW - dstW, 2 * dstW);
        iSrcOX += begin;

        for ( int32_t dstX = begin; dstX < end; ++dstX ) {
            //      srcOX = floor((dstX+0.5) / scale - 0.5);
            int32_t srcOX = int32_t(*iSrcOX++);
            float   sum   = 0;

            if ( dstX < mainBegin ) {
                dst[dstX] = cvt_roundss_su8(src[0]);
                continue;
            }
            if ( mainEnd <= dstX ) {
                dst[dstX] = cvt_roundss_su8(src[srcW - 1]);
                continue;
            }

            ptrdiff_t iCoef = (dstX % kVecStepX) + (dstX / kVecStepX % m_NumUnrolledCoordsX * tableWidth);
            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                float   coef = coefs[iCoef];
                int32_t srcX = srcOX + i;
                sum   += src[srcX] * coef;
                iCoef += kVecStepX;
            }

            dst[dstX] = cvt_roundss_su8(sum);
        }
    }

    //! resize horizontal (main loop)
    //!
    //! @param src    A row of source
    //! @param dst    A row of destination
    //! @param begin  Position of a first pixel
    //! @param end    Position of next of a last pixel
    void LinearResizerImpl<ArchAVX2FMA>::resizeXmain(
        const float * src, uint8_t * __restrict dst,
        int32_t begin, int32_t end
    ) {
        const float * coefs = &m_TablesX[0];
        const int32_t * indices = &m_IndicesX[0];
        ptrdiff_t tableWidth = ptrdiff_t(m_TablesXWidth);
        ptrdiff_t tableSize = tableWidth * m_NumUnrolledCoordsX;
        int32_t numCoefsX = m_NumCoefsX;

        ptrdiff_t iCoef = begin / kVecStepX % m_NumUnrolledCoordsX * tableWidth;
        for ( int32_t dstX = begin; dstX < end; dstX += kVecStepX ) {
            //      nume        = 0;
            __m256  f32x8Nume0  = _mm256_setzero_ps();
            __m256  f32x8Nume8  = _mm256_setzero_ps();
            //      srcOX       = floor(dstX / scale) + 1;
            __m256i s32x8SrcOX0 = _mm256_loadu_si256((const __m256i*)&indices[dstX + 0]);
            __m256i s32x8SrcOX8 = _mm256_loadu_si256((const __m256i*)&indices[dstX + 8]);

            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                //      srcX        = srcOX + i;
                __m256i s32x8Offset = _mm256_set1_epi32(i);
                __m256i s32x8SrcX0  = _mm256_add_epi32(s32x8SrcOX0, s32x8Offset);
                __m256i s32x8SrcX8  = _mm256_add_epi32(s32x8SrcOX8, s32x8Offset);

                //      nume        += src[srcX] * coefs[iCoef];
                __m256  s32x8Src0    = _mm256_i32gather_ps(src, s32x8SrcX0, sizeof(float));
                __m256  s32x8Src8    = _mm256_i32gather_ps(src, s32x8SrcX8, sizeof(float));
                __m256  f32x8Coefs0  = _mm256_load_ps(&coefs[iCoef + 0]);
                __m256  f32x8Coefs8  = _mm256_load_ps(&coefs[iCoef + 8]);
                f32x8Nume0 = _mm256_fmadd_ps(s32x8Src0, f32x8Coefs0, f32x8Nume0);
                f32x8Nume8 = _mm256_fmadd_ps(s32x8Src8, f32x8Coefs8, f32x8Nume8);

                iCoef += kVecStepX;
            }

            // dst[dstX] = clamp<int>(0, 255, round(nume));
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
    bool LinearResizerImpl_hasFeature<ArchAVX2FMA>()
    {
        return false;
    }

    template<>
    ILinearResizerImpl * LinearResizerImpl_new<ArchAVX2FMA>()
    {
        return NULL;
    }

}

#endif