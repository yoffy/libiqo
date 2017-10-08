#include "IQOAreaResizerImpl.hpp"


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
            int32_t dstW, float * dst,
            int32_t srcOY,
            const float * coefs
        );
        void resizeX(const float * src, uint8_t * dst);
        void resizeXmain(const float * src, uint8_t * dst);

        enum {
            //! for SIMD
            kVecStepX  = 16, //!< float32x4 x 4
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
        std::vector<float> m_TablesX_;  //!< Area table * m_NumCoordsX (unrolled)
        float * m_TablesX;              //!< aligned
        std::vector<float> m_TablesY;   //!< Area table * m_NumCoordsY
        std::vector<int32_t> m_IndicesX;
        std::vector<float> m_Work;
    };

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
        m_NumCoefsX = int32_t(calcNumCoefsForArea(rSrcW, rDstW));
        m_NumCoefsY = int32_t(calcNumCoefsForArea(rSrcH, rDstH));
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

        std::vector<float> tablesX(m_NumCoefsX * m_NumCoordsX);
        for ( int32_t dstX = 0; dstX < m_NumCoordsX; ++dstX ) {
            float * table = &tablesX[dstX * m_NumCoefsX];
            float sumCoefs = setAreaTable(rSrcW, rDstW, dstX, m_NumCoefsX, table);
            for ( int32_t i = 0; i < m_NumCoefsX; ++i ) {
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

        for ( int32_t dstY = 0; dstY < m_NumCoordsY; ++dstY ) {
            float * table = &m_TablesY[dstY * m_NumCoefsY];
            float sumCoefs = setAreaTable(rSrcH, rDstH, dstY, m_NumCoefsY, table);
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
            int32_t srcOX = int32_t(int64_t(dstX) * rSrcW / rDstW);
            m_IndicesX[dstX] = srcOX;
        }
    }

    void AreaResizerImpl<ArchSSE4_1>::resize(
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

        // main loop
#pragma omp parallel for
        for ( int32_t dstY = 0; dstY < dstH; ++dstY ) {
            float * work = &m_Work[HWCap::getThreadNumber() * ptrdiff_t(srcW)];
            int32_t srcOY = int32_t(int64_t(dstY) * srcH / dstH);
            const float * coefs = &tablesY[dstY % m_NumCoordsY * ptrdiff_t(m_NumCoefsY)];
            resizeYmain(
                srcSt, &src[0],
                srcW, work,
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
        int32_t dstW, float * __restrict dst,
        int32_t srcOY,
        const float * coefs
    ) {
        int32_t numCoefsY = m_NumCoefsY;
        int32_t vecLen = alignFloor<int32_t>(dstW, kVecStepY);

        for ( int32_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            //     nume         = 0;
            __m128 f32x4Nume0   = _mm_setzero_ps();
            __m128 f32x4Nume1   = _mm_setzero_ps();
            __m128 f32x4Nume2   = _mm_setzero_ps();
            __m128 f32x4Nume3   = _mm_setzero_ps();

            for ( int32_t i = 0; i < numCoefsY; ++i ) {
                int32_t srcY        = srcOY + i;

                //      coef        = coefs[i];
                __m128  f32x4Coef   = _mm_set1_ps(coefs[i]);

                //      nume       += src[dstX + srcSt*srcY] * coef;
                __m128i u8x16Src    = _mm_loadu_si128((const __m128i *)&src[dstX + srcSt*srcY]);
                _mm_prefetch((const char *)&src[dstX + srcSt*(srcY + numCoefsY)], _MM_HINT_T2);
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
                int32_t srcY = srcOY + i;
                float   coef = coefs[i];
                nume += src[dstX + srcSt * srcY] * coef;
            }

            dst[dstX] = nume;
        }
    }

    void AreaResizerImpl<ArchSSE4_1>::resizeX(const float * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            int32_t dstW = m_DstW;
            for ( int32_t dstX = 0; dstX < dstW; dstX++ ) {
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
        ptrdiff_t tableWidth = ptrdiff_t(m_TablesXWidth);
        ptrdiff_t tableSize = tableWidth * m_NumUnrolledCoordsX;
        int32_t numCoefsX = m_NumCoefsX;
        int32_t dstW = m_DstW;
        int32_t vecLen = alignFloor<int32_t>(dstW, kVecStepX);

        ptrdiff_t iCoef = 0;
        for ( int32_t dstX = 0; dstX < vecLen; dstX += kVecStepX ) {
            //      nume        = 0;
            __m128  f32x4Nume0  = _mm_setzero_ps();
            __m128  f32x4Nume1  = _mm_setzero_ps();
            __m128  f32x4Nume2  = _mm_setzero_ps();
            __m128  f32x4Nume3  = _mm_setzero_ps();
            //      srcOX       = floor(dstX / scale);
            __m128i s32x4SrcX0  = _mm_loadu_si128((const __m128i*)&indices[dstX +  0]);
            __m128i s32x4SrcX1  = _mm_loadu_si128((const __m128i*)&indices[dstX +  4]);
            __m128i s32x4SrcX2  = _mm_loadu_si128((const __m128i*)&indices[dstX +  8]);
            __m128i s32x4SrcX3  = _mm_loadu_si128((const __m128i*)&indices[dstX + 12]);

            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                //      nume       += src[srcX] * coefs[iCoef];
                __m128  f32x4Src0   = gather_ps(src, s32x4SrcX0);
                __m128  f32x4Src1   = gather_ps(src, s32x4SrcX1);
                __m128  f32x4Src2   = gather_ps(src, s32x4SrcX2);
                __m128  f32x4Src3   = gather_ps(src, s32x4SrcX3);
                __m128  f32x4Coefs0 = _mm_load_ps(&coefs[iCoef +  0]);
                __m128  f32x4Coefs1 = _mm_load_ps(&coefs[iCoef +  4]);
                __m128  f32x4Coefs2 = _mm_load_ps(&coefs[iCoef +  8]);
                __m128  f32x4Coefs3 = _mm_load_ps(&coefs[iCoef + 12]);
                __m128  f32x4iNume0 = _mm_mul_ps(f32x4Src0, f32x4Coefs0);
                __m128  f32x4iNume1 = _mm_mul_ps(f32x4Src1, f32x4Coefs1);
                __m128  f32x4iNume2 = _mm_mul_ps(f32x4Src2, f32x4Coefs2);
                __m128  f32x4iNume3 = _mm_mul_ps(f32x4Src3, f32x4Coefs3);
                f32x4Nume0 = _mm_add_ps(f32x4Nume0, f32x4iNume0);
                f32x4Nume1 = _mm_add_ps(f32x4Nume1, f32x4iNume1);
                f32x4Nume2 = _mm_add_ps(f32x4Nume2, f32x4iNume2);
                f32x4Nume3 = _mm_add_ps(f32x4Nume3, f32x4iNume3);

                //      srcX        = srcOX + i;
                const __m128i k1 = _mm_set1_epi32(1);
                s32x4SrcX0 = _mm_add_epi32(s32x4SrcX0, k1);
                s32x4SrcX1 = _mm_add_epi32(s32x4SrcX1, k1);
                s32x4SrcX2 = _mm_add_epi32(s32x4SrcX2, k1);
                s32x4SrcX3 = _mm_add_epi32(s32x4SrcX3, k1);

                iCoef += kVecStepX;
            }

            // dst[dstX] = clamp<int>(0, 255, round(nume));
            __m128i u8x8Dst0 = cvt_roundps_epu8(f32x4Nume0, f32x4Nume1);
            __m128i u8x8Dst1 = cvt_roundps_epu8(f32x4Nume2, f32x4Nume3);
            __m128i u8x16Dst = _mm_unpacklo_epi64(u8x8Dst0, u8x8Dst1);
            _mm_storeu_si128((__m128i*)&dst[dstX], u8x16Dst);

            // iCoef = dstX % tableSize;
            if ( iCoef == tableSize ) {
                iCoef = 0;
            }
        }

        for ( int32_t dstX = vecLen; dstX < dstW; ++dstX ) {
            int32_t srcOX = indices[dstX];
            float sum = 0;

            // calc index of coefs from unrolled table
            iCoef = (dstX % kVecStepX) + (dstX / kVecStepX % m_NumUnrolledCoordsX * tableWidth);

            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                int32_t srcX = srcOX + i;
                sum   += src[srcX] * coefs[iCoef];
                iCoef += kVecStepX;
            }

            dst[dstX] = cvt_roundss_su8(sum);
        }
    }

}

#else

namespace iqo {

    template<>
    IAreaResizerImpl * AreaResizerImpl_new<ArchSSE4_1>()
    {
        return NULL;
    }

}

#endif
