#include "IQOAreaResizerImpl.hpp"


#if defined(IQO_CPU_X86) && defined(IQO_HAVE_SSE4_1)

#include <cstring>
#include <vector>
#include <smmintrin.h>

#if defined(_OPENMP)
    #include <omp.h>
#endif

#include "math.hpp"
#include "util.hpp"
#include "IQOHWCap.hpp"


namespace {

    //! (uint8_t)min(255, max(0, round(v))) in Q7
    uint8_t cvt_roundu16q7_su8(uint16_t v)
    {
        uint16_t k0_5 = 1 << 6; // 0.5 in Q7
        return uint8_t((v + k0_5) >> 7);
    }

    //! (uint8_t)min(255, max(0, round(v))) in Q7
    __m128i cvt_roundu16q7_epu8(__m128i lo, __m128i hi)
    {
        __m128i u16x8k0_5 = _mm_set1_epi16(1 << 6); // 0.5 in Q7

        // v = (x + k0_5) >> 7
        __m128i u16x8v0 = _mm_srli_epi16(_mm_add_epi16(lo, u16x8k0_5), 7);
        __m128i u16x8v1 = _mm_srli_epi16(_mm_add_epi16(hi, u16x8k0_5), 7);

        // return (uint8_t)v
        return _mm_packus_epi16(u16x8v0, u16x8v1);
    }

    //! f32x4Dst[dstField] = srcPtr[s32x4Indices[srcField]]
    __m128i gather_epu16(const uint16_t * u16Src, __m128i u16x8Indices)
    {
        // MOVQ is very faster than PEXTRD/Q in Silvermont
        uint64_t u16x4Indices0 = _mm_cvtsi128_si64(u16x8Indices);
        uint64_t u16x4Indices1 = _mm_extract_epi64(u16x8Indices, 1);
        ptrdiff_t i0 = uint16_t(u16x4Indices0);
        ptrdiff_t i1 = uint16_t(u16x4Indices0 >> 16);
        ptrdiff_t i2 = uint16_t(u16x4Indices0 >> 32);
        ptrdiff_t i3 = uint16_t(u16x4Indices0 >> 48);
        ptrdiff_t i4 = uint16_t(u16x4Indices1);
        ptrdiff_t i5 = uint16_t(u16x4Indices1 >> 16);
        ptrdiff_t i6 = uint16_t(u16x4Indices1 >> 32);
        ptrdiff_t i7 = uint16_t(u16x4Indices1 >> 48);
        __m128i u16x8V = _mm_setzero_si128();
        u16x8V = _mm_insert_epi16(u16x8V, u16Src[i0], 0);
        u16x8V = _mm_insert_epi16(u16x8V, u16Src[i1], 1);
        u16x8V = _mm_insert_epi16(u16x8V, u16Src[i2], 2);
        u16x8V = _mm_insert_epi16(u16x8V, u16Src[i3], 3);
        u16x8V = _mm_insert_epi16(u16x8V, u16Src[i4], 4);
        u16x8V = _mm_insert_epi16(u16x8V, u16Src[i5], 5);
        u16x8V = _mm_insert_epi16(u16x8V, u16Src[i6], 6);
        u16x8V = _mm_insert_epi16(u16x8V, u16Src[i7], 7);
        return u16x8V;
    }

}

namespace iqo {

    template<>
    class AreaResizerImpl<ArchSSE4_1> : public IAreaResizerImpl
    {
    public:
        //! Constructor
        AreaResizerImpl();
        //! Destructor
        virtual ~AreaResizerImpl();

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
        //! dst[i] = src[i] * kBias / srcSum (src will be broken)
        void adjustCoefs(
            float * srcBegin, float * srcEnd,
            float srcSum,
            uint16_t bias,
            uint16_t * dst
        );

        void resizeYmain(
            ptrdiff_t srcSt, const uint8_t * src,
            int32_t dstW, uint16_t * dst,
            int32_t srcOY,
            const uint16_t * coefs
        );
        void resizeX(const uint16_t * src, uint8_t * dst);
        void resizeXmain(const uint16_t * src, uint8_t * dst);

        enum {
            kBiasXBit   = 15,
            kBiasX      = 1 << kBiasXBit,
            kBiasYBit   = 7, // it can be 8, but mulhrs is signed
            kBiasY      = 1 << kBiasYBit,

            //! for SIMD
            kVecStepX  = 16, //!< uint16x8 x 2
            kVecStepY  = 16, //!< uint16x8 x 2
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
        std::vector<uint16_t> m_TablesX_;   //!< Area table * m_NumCoordsX (unrolled)
        uint16_t * m_TablesX;               //!< aligned
        std::vector<uint16_t> m_TablesY;    //!< Area table * m_NumCoordsY
        std::vector<uint16_t> m_IndicesX;
        int32_t m_WorkW;                    //!< width of m_Work
        uint16_t * m_Work;
    };

    template<>
    IAreaResizerImpl * AreaResizerImpl_new<ArchSSE4_1>()
    {
        return new AreaResizerImpl<ArchSSE4_1>();
    }


    //! Constructor
    AreaResizerImpl<ArchSSE4_1>::AreaResizerImpl()
    {
        m_Work = NULL;
    }
    // Destructor
    AreaResizerImpl<ArchSSE4_1>::~AreaResizerImpl()
    {
        alignedFree(m_Work);
    }

    // Construct
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
        m_TablesX = reinterpret_cast<uint16_t *>(addrTablesX);
        size_t tblYSize = size_t(m_NumCoefsY) * m_NumCoordsY;
        m_TablesY.reserve(tblYSize);
        m_TablesY.resize(tblYSize);

        std::vector<uint16_t> tablesX(m_NumCoefsX * m_NumCoordsX);
        std::vector<float>    coefsX(m_NumCoefsX);
        for ( int32_t dstX = 0; dstX < m_NumCoordsX; ++dstX ) {
            uint16_t * table = &tablesX[dstX * m_NumCoefsX];
            float sumCoefs = setAreaTable(rSrcW, rDstW, dstX, m_NumCoefsX, &coefsX[0]);
            adjustCoefs(&coefsX[0], &coefsX[m_NumCoefsX], sumCoefs, kBiasX, &table[0]);
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

        std::vector<float>    coefsY(m_NumCoefsY);
        for ( int32_t dstY = 0; dstY < m_NumCoordsY; ++dstY ) {
            uint16_t * table = &m_TablesY[dstY * m_NumCoefsY];
            float sumCoefs = setAreaTable(rSrcH, rDstH, dstY, m_NumCoefsY, &coefsY[0]);
            adjustCoefs(&coefsY[0], &coefsY[m_NumCoefsY], sumCoefs, kBiasY, &table[0]);
        }

        // allocate workspace
        m_WorkW = alignCeil<int32_t>(m_SrcW, kVecStepY);
        int32_t workSize = m_WorkW * HWCap::getNumberOfProcs();
        m_Work = alignedAlloc<uint16_t>(workSize, sizeof(*m_Work) * kVecStepY);

        // calc indices
        int32_t alignedDstW = alignCeil<int32_t>(m_DstW, kVecStepX);
        m_IndicesX.reserve(alignedDstW);
        m_IndicesX.resize(alignedDstW);
        for ( int32_t dstX = 0; dstX < alignedDstW; ++dstX ) {
            //      srcOX = floor(dstX / scale)
            uint16_t srcOX = uint16_t(int64_t(dstX) * rSrcW / rDstW);
            m_IndicesX[dstX] = srcOX;
        }
    }

    void AreaResizerImpl<ArchSSE4_1>::adjustCoefs(
        float * __restrict srcBegin, float * __restrict srcEnd,
        float srcSum,
        uint16_t bias,
        uint16_t * __restrict dst)
    {
        const int k1_0 = bias;
        size_t numCoefs = srcEnd - srcBegin;
        int dstSum = 0;

        for ( size_t i = 0; i < numCoefs; ++i ) {
            dst[i] = uint16_t(round(srcBegin[i] * bias / srcSum));
            dstSum += dst[i];
        }
        while ( dstSum < k1_0 ) {
            size_t i = std::distance(&srcBegin[0], std::max_element(&srcBegin[0], &srcBegin[numCoefs]));
            dst[i]++;
            srcBegin[i] = 0;
            dstSum++;
        }
        while ( dstSum > k1_0 ) {
            size_t i = std::distance(&srcBegin[0], std::max_element(&srcBegin[0], &srcBegin[numCoefs]));
            dst[i]--;
            srcBegin[i] = 0;
            dstSum--;
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
                uint16_t * work = &m_Work[HWCap::getThreadNumber() * ptrdiff_t(m_WorkW)];
                for ( int32_t x = 0; x < srcW; ++x ) {
                    work[x] = src[srcSt * y + x];
                }
                resizeX(work, &dst[dstSt * y]);
            }
            return;
        }

        const uint16_t * tablesY = &m_TablesY[0];

        // main loop
#pragma omp parallel for
        for ( int32_t dstY = 0; dstY < dstH; ++dstY ) {
            uint16_t * work = &m_Work[HWCap::getThreadNumber() * ptrdiff_t(m_WorkW)];
            int32_t srcOY = int32_t(int64_t(dstY) * srcH / dstH);
            const uint16_t * coefs = &tablesY[dstY % m_NumCoordsY * ptrdiff_t(m_NumCoefsY)];
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
        int32_t dstW, uint16_t * __restrict dst,
        int32_t srcOY,
        const uint16_t * coefs
    ) {
        int32_t numCoefsY = m_NumCoefsY;
        int32_t vecLen = alignFloor<int32_t>(dstW, kVecStepY);

#if defined(__GNUC__)
        dst = reinterpret_cast<uint16_t *>(__builtin_assume_aligned(dst, sizeof(*dst)*kVecStepY));
#endif

        for ( int32_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            //     nume         = 0;
            __m128i u16x8Nume0   = _mm_setzero_si128();
            __m128i u16x8Nume1   = _mm_setzero_si128();

            for ( int32_t i = 0; i < numCoefsY; ++i ) {
                int32_t srcY        = srcOY + i;

                //      coef        = coefs[i];
                __m128i u16x8Coef   = _mm_set1_epi16(coefs[i]);

                //      nume       += src[dstX + srcSt*srcY] * coef;
                __m128i u8x16Src    = _mm_loadu_si128((const __m128i *)&src[dstX + srcSt*srcY]);
                _mm_prefetch((const char *)&src[dstX + srcSt*(srcY + numCoefsY)], _MM_HINT_T2);
                __m128i u16x8Src0   = _mm_unpacklo_epi8(u8x16Src, _mm_setzero_si128());
                __m128i u16x8Src1   = _mm_unpackhi_epi8(u8x16Src, _mm_setzero_si128());
                u16x8Nume0 = _mm_add_epi16(_mm_mullo_epi16(u16x8Src0, u16x8Coef), u16x8Nume0);
                u16x8Nume1 = _mm_add_epi16(_mm_mullo_epi16(u16x8Src1, u16x8Coef), u16x8Nume1);
            }

            // dst[dstX] = nume;
            __m128i u16x8Dst0 = u16x8Nume0;
            __m128i u16x8Dst1 = u16x8Nume1;
            _mm_store_si128((__m128i *)&dst[dstX + 0], u16x8Dst0);
            _mm_store_si128((__m128i *)&dst[dstX + 8], u16x8Dst1);
        }

        for ( int32_t dstX = vecLen; dstX < dstW; dstX++ ) {
            uint16_t nume = 0;

            for ( int32_t i = 0; i < numCoefsY; ++i ) {
                int32_t  srcY = srcOY + i;
                uint16_t coef = coefs[i];
                nume += src[dstX + srcSt * srcY] * coef;
            }

            dst[dstX] = nume;
        }
    }

    void AreaResizerImpl<ArchSSE4_1>::resizeX(const uint16_t * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            int32_t dstW = m_DstW;
            for ( int32_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = cvt_roundu16q7_su8(src[dstX]);
            }
            return;
        }

        resizeXmain(src, dst);
    }

    //! resize horizontal (main loop)
    //!
    //! @param src    A row of source
    //! @param dst    A row of destination
    void AreaResizerImpl<ArchSSE4_1>::resizeXmain(const uint16_t * src, uint8_t * __restrict dst)
    {
        const uint16_t * coefs = &m_TablesX[0];
        const uint16_t * indices = &m_IndicesX[0];
        ptrdiff_t tableWidth = ptrdiff_t(m_TablesXWidth);
        ptrdiff_t tableSize = tableWidth * m_NumUnrolledCoordsX;
        int32_t numCoefsX = m_NumCoefsX;
        int32_t dstW = m_DstW;
        int32_t vecLen = alignFloor<int32_t>(dstW, kVecStepX);

        ptrdiff_t iCoef = 0;
        for ( int32_t dstX = 0; dstX < vecLen; dstX += kVecStepX ) {
            //      nume        = 0;
            __m128i u16x8Nume0  = _mm_setzero_si128();
            __m128i u16x8Nume1  = _mm_setzero_si128();
            //      srcOX       = floor(dstX / scale);
            __m128i u16x8SrcX0  = _mm_loadu_si128((const __m128i*)&indices[dstX + 0]);
            __m128i u16x8SrcX1  = _mm_loadu_si128((const __m128i*)&indices[dstX + 8]);

            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                //      nume       += src[srcX] * coefs[iCoef];
                __m128i u16x8Src0   = gather_epu16(src, u16x8SrcX0);
                __m128i u16x8Src1   = gather_epu16(src, u16x8SrcX1);
                __m128i u16x8Coefs0 = _mm_load_si128((const __m128i *)&coefs[iCoef + 0]);
                __m128i u16x8Coefs1 = _mm_load_si128((const __m128i *)&coefs[iCoef + 8]);
                //                    (src*kBiasY * coef*kBiasX) / kBiasX
                __m128i u16x8iNume0 = _mm_mulhrs_epi16(u16x8Src0, u16x8Coefs0);
                __m128i u16x8iNume1 = _mm_mulhrs_epi16(u16x8Src1, u16x8Coefs1);
                u16x8Nume0 = _mm_add_epi16(u16x8Nume0, u16x8iNume0);
                u16x8Nume1 = _mm_add_epi16(u16x8Nume1, u16x8iNume1);

                //      srcX        = srcOX + i;
                const __m128i k1 = _mm_set1_epi16(1);
                u16x8SrcX0 = _mm_add_epi16(u16x8SrcX0, k1);
                u16x8SrcX1 = _mm_add_epi16(u16x8SrcX1, k1);

                iCoef += kVecStepX;
            }

            // dst[dstX] = clamp<int>(0, 255, round(nume));
            __m128i u8x16Dst = cvt_roundu16q7_epu8(u16x8Nume0, u16x8Nume1);
            _mm_storeu_si128((__m128i *)&dst[dstX], u8x16Dst);

            // iCoef = dstX % tableSize;
            if ( iCoef == tableSize ) {
                iCoef = 0;
            }
        }

        for ( int32_t dstX = vecLen; dstX < dstW; ++dstX ) {
            int32_t srcOX = indices[dstX];
            uint16_t sum = 0;

            // calc index of coefs from unrolled table
            iCoef = (dstX % kVecStepX) + (dstX / kVecStepX % m_NumUnrolledCoordsX * tableWidth);

            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                int32_t srcX = srcOX + i;
                sum   += (src[srcX] * coefs[iCoef]) / kBiasX;
                iCoef += kVecStepX;
            }

            dst[dstX] = cvt_roundu16q7_su8(sum);
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
