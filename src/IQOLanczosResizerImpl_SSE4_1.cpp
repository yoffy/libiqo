#include "IQOLanczosResizerImpl.hpp"


#if defined(IQO_CPU_X86) && defined(IQO_HAVE_SSE4_1)

#include <cstring>
#include <vector>
#include <smmintrin.h>

#if defined(_OPENMP)
    #include <omp.h>
#endif

#include "math.hpp"
#include "IQOHWCap.hpp"


namespace {

    //! f32x4Dst[dstField] = f32x4Src[0]
    template<int dstField>
    __m128 insert_ps(__m128 f32x4Dst, __m128 f32x4Src)
    {
#if defined(__GNUC__)
        // separate loading and insertion for performance (33% faster on Cherry Trail)
        __asm__ (
            "insertps $%c[field], %[src], %[dst]  \n\t"
            : [dst]"=x"(f32x4Dst)
            : [dst]"x"(f32x4Dst), [src]"x"(f32x4Src), [field]"i"(dstField << 4)
            :
        );
        return f32x4Dst;
#else
        return _mm_insert_ps(f32x4Dst, f32x4Src, dstField << 4);
#endif
    }

    //! f32x4Dst[dstField] = srcPtr[s32x4Indices[srcField]]
    __m128 gather_ps(const float * f32Src, __m128i s32x4Indices)
    {
        // MOVQ is very faster than PEXTRD/Q in Silvermont
        uint64_t s32x2Indices0 = _mm_cvtsi128_si64(s32x4Indices);
        uint64_t s32x2Indices1 = _mm_extract_epi64(s32x4Indices, 1);
        ptrdiff_t i0 = int32_t(s32x2Indices0);
        ptrdiff_t i1 = int32_t(s32x2Indices0 >> 32);
        ptrdiff_t i2 = int32_t(s32x2Indices1);
        ptrdiff_t i3 = int32_t(s32x2Indices1 >> 32);
        __m128 f32x4V = _mm_load_ss(&f32Src[i0]);
        f32x4V = insert_ps<1>(f32x4V, _mm_load_ss(&f32Src[i1]));
        f32x4V = insert_ps<2>(f32x4V, _mm_load_ss(&f32Src[i2]));
        f32x4V = insert_ps<3>(f32x4V, _mm_load_ss(&f32Src[i3]));
        return f32x4V;
    }

    __m128 mask_gather_ps(const float * f32Src, __m128i s32x4Indices, __m128i u32x4Mask)
    {
        // MOVQ is very faster than PEXTRD/Q in Silvermont
        uint64_t s32x2Indices0 = _mm_cvtsi128_si64(s32x4Indices);
        uint64_t s32x2Indices1 = _mm_extract_epi64(s32x4Indices, 1);
        ptrdiff_t i0 = int32_t(s32x2Indices0);
        ptrdiff_t i1 = int32_t(s32x2Indices0 >> 32);
        ptrdiff_t i2 = int32_t(s32x2Indices1);
        ptrdiff_t i3 = int32_t(s32x2Indices1 >> 32);
        uint32_t b16Mask = _mm_movemask_epi8(u32x4Mask);
        __m128 f32x4V = _mm_setzero_ps();
        if ( b16Mask & 0x0001 ) f32x4V = _mm_load_ss(&f32Src[i0]);
        if ( b16Mask & 0x0010 ) f32x4V = insert_ps<1>(f32x4V, _mm_load_ss(&f32Src[i1]));
        if ( b16Mask & 0x0100 ) f32x4V = insert_ps<2>(f32x4V, _mm_load_ss(&f32Src[i2]));
        if ( b16Mask & 0x1000 ) f32x4V = insert_ps<3>(f32x4V, _mm_load_ss(&f32Src[i3]));
        return f32x4V;
    }

    uint8_t cvt_roundss_su8(float v)
    {
        __m128  f32x1V     = _mm_set_ss(v);
        __m128  f32x1Round = _mm_round_ss(f32x1V, f32x1V, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        int32_t s32x1Round = _mm_cvtss_si32(f32x1Round);
        return uint8_t(iqo::clamp(0, 255, s32x1Round));
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
    class LanczosResizerImpl<ArchSSE4_1> : public ILanczosResizerImpl
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
            int32_t dstW, float * dst,
            int32_t srcOY,
            const float * coefs
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
            int32_t begin, int32_t end
        );
        void resizeXmain(
            const float * src, uint8_t * dst,
            int32_t begin, int32_t end
        );

        enum {
            //! for SIMD
            kVecStepX  =  8, //!< float32x4 x 2
            kVecStepY  = 16, //!< float32x4 x 4
        };
        int32_t m_SrcW;
        int32_t m_SrcH;
        int32_t m_DstW;
        int32_t m_DstH;
        int32_t m_NumCoefsX;
        int32_t m_NumCoefsY;
        int32_t m_NumCoordsX;
        int32_t m_NumUnrolledCoordsX;
        int32_t m_TablesXWidth;
        int32_t m_NumCoordsY;
        std::vector<float> m_TablesX_;  //!< m_TablesXWidth * m_NumCoordsX (unrolled)
        float * m_TablesX;              //!< aligned
        std::vector<float> m_TablesY;   //!< Lanczos table * m_NumCoordsY
        std::vector<int32_t> m_IndicesX;
        std::vector<float> m_Work;
    };

    template<>
    ILanczosResizerImpl * LanczosResizerImpl_new<ArchSSE4_1>()
    {
        return new LanczosResizerImpl<ArchSSE4_1>();
    }


    // Constructor
    void LanczosResizerImpl<ArchSSE4_1>::init(
        unsigned int degree,
        size_t srcW, size_t srcH,
        size_t dstW, size_t dstH,
        size_t pxScale
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
        m_NumCoefsX = int32_t(calcNumCoefsForLanczos(degree, rSrcW, rDstW, pxScale));
        m_NumCoefsY = int32_t(calcNumCoefsForLanczos(degree, rSrcH, rDstH, pxScale));
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
        for ( int32_t dstX = 0; dstX < m_NumCoordsX; ++dstX ) {
            float * table = &tablesX[dstX * m_NumCoefsX];
            float sumCoefs = setLanczosTable(degree, rSrcW, rDstW, dstX, pxScale, m_NumCoefsX, table);
            for ( int32_t i = 0; i < m_NumCoefsX; i++ ) {
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
        for ( int32_t dstY = 0; dstY < m_NumCoordsY; ++dstY ) {
            float * table = &m_TablesY[dstY * m_NumCoefsY];
            float sumCoefs = setLanczosTable(degree, rSrcH, rDstH, dstY, pxScale, m_NumCoefsY, table);
            for ( int32_t i = 0; i < m_NumCoefsY; ++i ) {
                table[i] /= sumCoefs;
            }
        }

        // allocate workspace
        m_Work.reserve(m_SrcW * HWCap::getNumberOfProcs());
        m_Work.resize(m_SrcW * HWCap::getNumberOfProcs());

        // calc indices
        int32_t alignedDstW = alignCeil<int32_t>(m_DstW, kVecStepX);
        m_IndicesX.reserve(alignedDstW);
        m_IndicesX.resize(alignedDstW);
        for ( int32_t dstX = 0; dstX < alignedDstW; ++dstX ) {
            //      srcOX = floor(dstX / scale)
            int32_t srcOX = int32_t(int64_t(dstX) * rSrcW / rDstW + 1);
            m_IndicesX[dstX] = srcOX;
        }
    }

    void LanczosResizerImpl<ArchSSE4_1>::resize(
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

        int64_t numCoefsOn2 = m_NumCoefsY / 2;
        // mainBegin = std::ceil((numCoefsOn2 - 1) * dstH / double(srcH))
        int32_t mainBegin = int32_t(((numCoefsOn2 - 1) * dstH + srcH-1) / srcH);
        int32_t mainEnd = std::max(0, int32_t((srcH - numCoefsOn2) * dstH / srcH));
        const float * tablesY = &m_TablesY[0];

        // border pixels
#pragma omp parallel for
        for ( ptrdiff_t dstY = 0; dstY < mainBegin; ++dstY ) {
            float * work = &m_Work[HWCap::getThreadNumber() * ptrdiff_t(srcW)];
            int32_t srcOY = int32_t(int64_t(dstY) * srcH / dstH + 1);
            const float * coefs = &tablesY[dstY % m_NumCoordsY * ptrdiff_t(m_NumCoefsY)];
            resizeYborder(
                srcSt, &src[0],
                srcW, work,
                srcOY,
                coefs);
            resizeX(work, &dst[dstSt * dstY]);
        }

        // main loop
#pragma omp parallel for
        for ( ptrdiff_t dstY = mainBegin; dstY < mainEnd; ++dstY ) {
            float * work = &m_Work[HWCap::getThreadNumber() * ptrdiff_t(srcW)];
            int32_t srcOY = int32_t(int64_t(dstY) * srcH / dstH + 1);
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
            int32_t srcOY = int32_t(int64_t(dstY) * srcH / dstH + 1);
            const float * coefs = &tablesY[dstY % m_NumCoordsY * ptrdiff_t(m_NumCoefsY)];
            resizeYborder(
                srcSt, &src[0],
                srcW, work,
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
    void LanczosResizerImpl<ArchSSE4_1>::resizeYborder(
        ptrdiff_t srcSt, const uint8_t * src,
        int32_t dstW, float * __restrict dst,
        int32_t srcOY,
        const float * coefs
    ) {
        int32_t numCoefsOn2 = m_NumCoefsY / 2;
        int32_t vecLen = alignFloor<int32_t>(dstW, kVecStepY);
        int32_t numCoefsY = m_NumCoefsY;
        int32_t srcH = m_SrcH;

        for ( int32_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            //     nume       = 0;
            __m128 f32x4Nume0 = _mm_setzero_ps();
            __m128 f32x4Nume1 = _mm_setzero_ps();
            __m128 f32x4Nume2 = _mm_setzero_ps();
            __m128 f32x4Nume3 = _mm_setzero_ps();
            float deno = 0;

            for ( int32_t i = 0; i < numCoefsY; ++i ) {
                int32_t srcY = srcOY - numCoefsOn2 + i;

                if ( 0 <= srcY && srcY < srcH ) {
                    //      coef        = coefs[i];
                    __m128  f32x4Coef   = _mm_set1_ps(coefs[i]);

                    //      nume       += src[dstX + srcSt*srcY] * coef;
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
                    deno += coefs[i];
                }
            }

            // dst[dstX] = nume / deno;
            // precision of RCPPS is only 11-bit, but it grater than precision of source image (8-bit).
            __m128 f32x4RcpDeno = _mm_rcp_ps(_mm_set1_ps(deno));
            __m128 f32x4Dst0    = _mm_mul_ps(f32x4Nume0, f32x4RcpDeno);
            __m128 f32x4Dst1    = _mm_mul_ps(f32x4Nume1, f32x4RcpDeno);
            __m128 f32x4Dst2    = _mm_mul_ps(f32x4Nume2, f32x4RcpDeno);
            __m128 f32x4Dst3    = _mm_mul_ps(f32x4Nume3, f32x4RcpDeno);
            _mm_storeu_ps(&dst[dstX +  0], f32x4Dst0);
            _mm_storeu_ps(&dst[dstX +  4], f32x4Dst1);
            _mm_storeu_ps(&dst[dstX +  8], f32x4Dst2);
            _mm_storeu_ps(&dst[dstX + 12], f32x4Dst3);
        }

        for ( int32_t dstX = vecLen; dstX < dstW; dstX++ ) {
            float nume = 0;
            float deno = 0;

            for ( int32_t i = 0; i < numCoefsY; ++i ) {
                int32_t srcY = srcOY - numCoefsOn2 + i;
                float   coef = coefs[i];
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
    void LanczosResizerImpl<ArchSSE4_1>::resizeYmain(
        ptrdiff_t srcSt, const uint8_t * src,
        int32_t dstW, float * __restrict dst,
        int32_t srcOY,
        const float * coefs
    ) {
        int32_t numCoefsOn2 = m_NumCoefsY / 2;
        int32_t vecLen = alignFloor<int32_t>(dstW, kVecStepY);
        int32_t numCoefsY = m_NumCoefsY;

        for ( int32_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            //     nume       = 0;
            __m128 f32x4Nume0 = _mm_setzero_ps();
            __m128 f32x4Nume1 = _mm_setzero_ps();
            __m128 f32x4Nume2 = _mm_setzero_ps();
            __m128 f32x4Nume3 = _mm_setzero_ps();

            for ( int32_t i = 0; i < numCoefsY; ++i ) {
                int32_t srcY        = srcOY - numCoefsOn2 + i;

                //      coef        = coefs[i];
                __m128  f32x4Coef   = _mm_set1_ps(coefs[i]);

                //      nume       += src[dstX + srcSt*srcY] * coef;
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

        for ( int32_t dstX = vecLen; dstX < dstW; dstX++ ) {
            float nume = 0;

            for ( int32_t i = 0; i < numCoefsY; ++i ) {
                int32_t srcY = srcOY - numCoefsOn2 + i;
                float   coef = coefs[i];
                nume += src[dstX + srcSt * srcY] * coef;
            }

            dst[dstX] = nume;
        }
    }

    void LanczosResizerImpl<ArchSSE4_1>::resizeX(const float * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            int32_t dstW = m_DstW;
            for ( int32_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = cvt_roundss_su8(src[dstX]);
            }
            return;
        }

        int64_t numCoefsOn2 = m_NumCoefsX / 2;
        // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstW / double(m_SrcW))
        int32_t mainBegin = int32_t(((numCoefsOn2 - 1) * m_DstW + m_SrcW-1) / m_SrcW);
        mainBegin = alignCeil<int32_t>(mainBegin, kVecStepX);
        int32_t mainEnd = std::max(0, int32_t((m_SrcW - numCoefsOn2) * m_DstW / m_SrcW));
        int32_t mainLen = alignFloor<int32_t>(mainEnd - mainBegin, kVecStepX);
        mainEnd = mainBegin + mainLen;

        resizeXborder(src, dst, 0,         mainBegin);
        resizeXmain  (src, dst, mainBegin, mainEnd);
        resizeXborder(src, dst, mainEnd,   m_DstW);
    }

    //! resize horizontal (border loop)
    //!
    //! @param src    A row of source
    //! @param dst    A row of destination
    //! @param begin  Position of a first pixel
    //! @param end    Position of next of a last pixel
    void LanczosResizerImpl<ArchSSE4_1>::resizeXborder(
        const float * src, uint8_t * __restrict dst,
        int32_t begin, int32_t end
    ) {
        int32_t numCoefsOn2 = m_NumCoefsX / 2;
        const float * coefs = &m_TablesX[0];
        const int32_t * indices = &m_IndicesX[0];
        ptrdiff_t tableWidth = ptrdiff_t(m_TablesXWidth);
        ptrdiff_t tableSize = tableWidth * m_NumUnrolledCoordsX;
        int32_t numCoefsX = m_NumCoefsX;

        const __m128i s32x4k0   = _mm_setzero_si128();
        const __m128i s32x4SrcW = _mm_set1_epi32(m_SrcW);
        ptrdiff_t iCoef = begin / kVecStepX % m_NumUnrolledCoordsX * tableWidth;
        for ( int32_t dstX = begin; dstX < end; dstX += kVecStepX ) {
            //      nume        = 0;
            __m128  f32x4Nume0  = _mm_setzero_ps();
            __m128  f32x4Nume1  = _mm_setzero_ps();
            //      deno        = 0;
            __m128  f32x4Deno0  = _mm_setzero_ps();
            __m128  f32x4Deno1  = _mm_setzero_ps();
            //      srcOX       = floor(dstX / scale) + 1;
            __m128i s32x4SrcOX0 = _mm_loadu_si128((const __m128i*)&indices[dstX + 0]);
            __m128i s32x4SrcOX1 = _mm_loadu_si128((const __m128i*)&indices[dstX + 4]);

            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                //      srcX        = srcOX - numCoefsOn2 + i;
                __m128i s32x4Offset = _mm_set1_epi32(i - numCoefsOn2);
                __m128i s32x4SrcX0  = _mm_add_epi32(s32x4SrcOX0, s32x4Offset);
                __m128i s32x4SrcX1  = _mm_add_epi32(s32x4SrcOX1, s32x4Offset);

                // if ( 0 <= srcX && srcX < m_SrcW )
                __m128i u32x4Mask0  = _mm_andnot_si128(
                    _mm_cmpgt_epi32(s32x4k0, s32x4SrcX0),   // ~(0 > srcX)
                    _mm_cmpgt_epi32(s32x4SrcW, s32x4SrcX0)  // srcW > srcX
                );
                __m128i u32x4Mask1  = _mm_andnot_si128(
                    _mm_cmpgt_epi32(s32x4k0, s32x4SrcX1),
                    _mm_cmpgt_epi32(s32x4SrcW, s32x4SrcX1)
                );

                //      iNume      += src[srcX] * coefs[iCoef];
                __m128  f32x4Src0   = mask_gather_ps(src, s32x4SrcX0, u32x4Mask0);
                __m128  f32x4Src1   = mask_gather_ps(src, s32x4SrcX1, u32x4Mask1);
                __m128  f32x4Coefs0 = _mm_load_ps(&coefs[iCoef + 0]);
                __m128  f32x4Coefs1 = _mm_load_ps(&coefs[iCoef + 4]);
                __m128  f32x4iNume0 = _mm_mul_ps(f32x4Src0, f32x4Coefs0);
                __m128  f32x4iNume1 = _mm_mul_ps(f32x4Src1, f32x4Coefs1);
                // nume   += iNume;
                f32x4Nume0 = _mm_add_ps(f32x4Nume0, f32x4iNume0);
                f32x4Nume1 = _mm_add_ps(f32x4Nume1, f32x4iNume1);
                // deno   += coefs[iCoef];
                f32x4Deno0 = _mm_add_ps(f32x4Deno0, f32x4Coefs0);
                f32x4Deno1 = _mm_add_ps(f32x4Deno1, f32x4Coefs1);

                iCoef += kVecStepX;
            }

            // dst[dstX] = clamp<int>(0, 255, round(nume / deno));
            // precision of RCPPS is only 11-bit, but it grater than precision of source image (8-bit).
            __m128  f32x4RcpDeno0 = _mm_rcp_ps(f32x4Deno0);
            __m128  f32x4RcpDeno1 = _mm_rcp_ps(f32x4Deno1);
            __m128  f32x4Dst0     = _mm_mul_ps(f32x4Nume0, f32x4RcpDeno0);
            __m128  f32x4Dst1     = _mm_mul_ps(f32x4Nume1, f32x4RcpDeno1);
            __m128i u8x8Dst       = cvt_roundps_epu8(f32x4Dst0, f32x4Dst1);
            _mm_storel_epi64((__m128i*)&dst[dstX], u8x8Dst);

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
    void LanczosResizerImpl<ArchSSE4_1>::resizeXmain(
        const float * src, uint8_t * __restrict dst,
        int32_t begin, int32_t end
    ) {
        int32_t numCoefsOn2 = m_NumCoefsX / 2;
        const float * coefs = &m_TablesX[0];
        const int32_t * indices = &m_IndicesX[0];
        ptrdiff_t tableWidth = ptrdiff_t(m_TablesXWidth);
        ptrdiff_t tableSize = tableWidth * m_NumUnrolledCoordsX;
        int32_t numCoefsX = m_NumCoefsX;

        ptrdiff_t iCoef = begin / kVecStepX % m_NumUnrolledCoordsX * tableWidth;
        for ( int32_t dstX = begin; dstX < end; dstX += kVecStepX ) {
            //      nume        = 0;
            __m128  f32x4Nume0  = _mm_setzero_ps();
            __m128  f32x4Nume1  = _mm_setzero_ps();
            //      srcOX       = floor(dstX / scale) + 1;
            __m128i s32x4SrcOX0 = _mm_loadu_si128((const __m128i*)&indices[dstX + 0]);
            __m128i s32x4SrcOX1 = _mm_loadu_si128((const __m128i*)&indices[dstX + 4]);

            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                //      srcX        = srcOX - numCoefsOn2 + i;
                __m128i s32x4Offset = _mm_set1_epi32(i - numCoefsOn2);
                __m128i s32x4SrcX0  = _mm_add_epi32(s32x4SrcOX0, s32x4Offset);
                __m128i s32x4SrcX1  = _mm_add_epi32(s32x4SrcOX1, s32x4Offset);

                //      iNume      += src[srcX] * coefs[iCoef];
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

            // dst[dstX] = clamp<int>(0, 255, round(nume));
            __m128  f32x4Dst0     = f32x4Nume0;
            __m128  f32x4Dst1     = f32x4Nume1;
            __m128i u8x8Dst       = cvt_roundps_epu8(f32x4Dst0, f32x4Dst1);
            _mm_storel_epi64((__m128i*)&dst[dstX], u8x8Dst);

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
    ILanczosResizerImpl * LanczosResizerImpl_new<ArchSSE4_1>()
    {
        return NULL;
    }

}

#endif
