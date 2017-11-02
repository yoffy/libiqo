#include "IQOAreaResizerImpl.hpp"


#if defined(IQO_CPU_X86) && defined(IQO_HAVE_AVX2FMA)

#include <cstring>
#include <vector>
#include <immintrin.h>

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
    __m256i cvt_roundu16q7_epu8(__m256i lo, __m256i hi)
    {
        __m256i u16x16k0_5 = _mm256_set1_epi16(1 << 6); // 0.5 in Q7

        // v = (x + k0_5) >> 7
        __m256i u16x16v0 = _mm256_srli_epi16(_mm256_add_epi16(lo, u16x16k0_5), 7);
        __m256i u16x16v1 = _mm256_srli_epi16(_mm256_add_epi16(hi, u16x16k0_5), 7);

        // Perm = (uint8_t)v
        __m256i u8x32Perm = _mm256_packus_epi16(u16x16v0, u16x16v1);
        return _mm256_permute4x64_epi64(u8x32Perm, _MM_SHUFFLE(3, 1, 2, 0));
    }

    //! returns { a.i128[I0], b.i128[I1] };
    template<int I0, int I1>
    __m256i permute2x128(__m256i a, __m256i b)
    {
        return _mm256_permute2x128_si256(a, b, (I1 << 4) | I0);
    }

    __m256i i32gather_epu16(const uint16_t * src, __m256i s32x8i0, __m256i s32x8i1)
    {
        const int32_t * p = reinterpret_cast<const int32_t *>(src);

        __m256i s32x8Mask   = _mm256_set1_epi32(0x0000FFFF);
        __m256i s32x8SrcL   = _mm256_i32gather_epi32(p, s32x8i0, 2); // higher 16bit are broken
        __m256i s32x8SrcH   = _mm256_i32gather_epi32(p, s32x8i1, 2);
        __m256i s16x8SrcL   = _mm256_and_si256(s32x8SrcL, s32x8Mask);
        __m256i s16x8SrcH   = _mm256_and_si256(s32x8SrcH, s32x8Mask);
        __m256i s16x16Pack  = _mm256_packus_epi32(s16x8SrcL, s16x8SrcH);
        __m256i s16x16Src   = _mm256_permute4x64_epi64(s16x16Pack, _MM_SHUFFLE(3, 1, 2, 0));
        return s16x16Src;
    }

}

namespace iqo {

    template<>
    class AreaResizerImpl<ArchAVX2FMA> : public IAreaResizerImpl
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
            kVecStepX   = 32, //!< uint16x16 x 2
            kVecStepY   = 32, //!< uint16x16 x 2
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
        std::vector<int32_t> m_IndicesX;
        int32_t m_WorkW;                    //!< width of m_Work
        uint16_t * m_Work;
    };

    template<>
    IAreaResizerImpl * AreaResizerImpl_new<ArchAVX2FMA>()
    {
        return new AreaResizerImpl<ArchAVX2FMA>();
    }


    //! Constructor
    AreaResizerImpl<ArchAVX2FMA>::AreaResizerImpl()
    {
        m_Work = NULL;
    }
    // Destructor
    AreaResizerImpl<ArchAVX2FMA>::~AreaResizerImpl()
    {
        alignedFree(m_Work);
    }

    // Construct
    void AreaResizerImpl<ArchAVX2FMA>::init(
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
            int32_t srcOX = int32_t(int64_t(dstX) * rSrcW / rDstW);
            m_IndicesX[dstX] = srcOX;
        }
    }

    void AreaResizerImpl<ArchAVX2FMA>::adjustCoefs(
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

    void AreaResizerImpl<ArchAVX2FMA>::resize(
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
    void AreaResizerImpl<ArchAVX2FMA>::resizeYmain(
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
            //      nume        = 0;
            __m256i u16x16Nume0 = _mm256_setzero_si256(); // permuted by unpack
            __m256i u16x16Nume1 = _mm256_setzero_si256();

            for ( int32_t i = 0; i < numCoefsY; ++i ) {
                int32_t srcY        = srcOY + i;

                //      coef        = coefs[i];
                __m256i u16x16Coef  = _mm256_set1_epi16(coefs[i]);

                //      nume       += src[dstX + srcSt*srcY] * coef;
                __m256i u8x32Src    = _mm256_loadu_si256((const __m256i *)&src[dstX + srcSt*srcY]);
                _mm_prefetch((const char *)&src[dstX + srcSt*(srcY + numCoefsY)], _MM_HINT_T2);
                // another solution: _mm256_cvtepu8_epi16
                __m256i u16x16Src0      = _mm256_unpacklo_epi8(u8x32Src, _mm256_setzero_si256());
                __m256i u16x16Src1      = _mm256_unpackhi_epi8(u8x32Src, _mm256_setzero_si256());
                __m256i u16x16iNume0    = _mm256_mullo_epi16(u16x16Src0, u16x16Coef);
                __m256i u16x16iNume1    = _mm256_mullo_epi16(u16x16Src1, u16x16Coef);
                u16x16Nume0 = _mm256_add_epi16(u16x16iNume0, u16x16Nume0);
                u16x16Nume1 = _mm256_add_epi16(u16x16iNume1, u16x16Nume1);
            }

            // dst[dstX] = nume;
            __m256i u16x16Dst0  = permute2x128<0, 2>(u16x16Nume0, u16x16Nume1);
            __m256i u16x16Dst1  = permute2x128<1, 3>(u16x16Nume0, u16x16Nume1);
            _mm256_store_si256((__m256i *)&dst[dstX +  0], u16x16Dst0);
            _mm256_store_si256((__m256i *)&dst[dstX + 16], u16x16Dst1);
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

    void AreaResizerImpl<ArchAVX2FMA>::resizeX(const uint16_t * src, uint8_t * __restrict dst)
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
    void AreaResizerImpl<ArchAVX2FMA>::resizeXmain(const uint16_t * src, uint8_t * __restrict dst)
    {
        const uint16_t * coefs = &m_TablesX[0];
        const int32_t * indices = &m_IndicesX[0];
        ptrdiff_t tableWidth = ptrdiff_t(m_TablesXWidth);
        ptrdiff_t tableSize = tableWidth * m_NumUnrolledCoordsX;
        int32_t numCoefsX = m_NumCoefsX;
        int32_t dstW = m_DstW;
        int32_t vecLen = alignFloor<int32_t>(dstW, kVecStepX);

        ptrdiff_t iCoef = 0;
        for ( int32_t dstX = 0; dstX < vecLen; dstX += kVecStepX ) {
            //      nume        = 0;
            __m256i u16x16Nume0 = _mm256_setzero_si256();
            __m256i u16x16Nume1 = _mm256_setzero_si256();
            //      srcOX       = floor(dstX / scale);
            __m256i s32x8SrcOX0 = _mm256_loadu_si256((const __m256i*)&indices[dstX +  0]);
            __m256i s32x8SrcOX1 = _mm256_loadu_si256((const __m256i*)&indices[dstX +  8]);
            __m256i s32x8SrcOX2 = _mm256_loadu_si256((const __m256i*)&indices[dstX + 16]);
            __m256i s32x8SrcOX3 = _mm256_loadu_si256((const __m256i*)&indices[dstX + 24]);

            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                //      srcX        = srcOX + i;
                __m256i s32x8Offset = _mm256_set1_epi32(i);
                __m256i s32x8SrcX0  = _mm256_add_epi32(s32x8SrcOX0, s32x8Offset);
                __m256i s32x8SrcX1  = _mm256_add_epi32(s32x8SrcOX1, s32x8Offset);
                __m256i s32x8SrcX2  = _mm256_add_epi32(s32x8SrcOX2, s32x8Offset);
                __m256i s32x8SrcX3  = _mm256_add_epi32(s32x8SrcOX3, s32x8Offset);

                //      nume           += src[srcX] * coefs[iCoef];
                __m256i u16x16Src0      = i32gather_epu16(src, s32x8SrcX0, s32x8SrcX1);
                __m256i u16x16Src1      = i32gather_epu16(src, s32x8SrcX2, s32x8SrcX3);
                __m256i u16x16Coefs0    = _mm256_load_si256((const __m256i *)&coefs[iCoef +  0]);
                __m256i u16x16Coefs1    = _mm256_load_si256((const __m256i *)&coefs[iCoef + 16]);
                //                        (src*kBiasY * coef*kBiasX) / kBiasX
                __m256i u16x16iNume0    = _mm256_mulhrs_epi16(u16x16Src0, u16x16Coefs0);
                __m256i u16x16iNume1    = _mm256_mulhrs_epi16(u16x16Src1, u16x16Coefs1);
                u16x16Nume0 = _mm256_add_epi16(u16x16iNume0, u16x16Nume0);
                u16x16Nume1 = _mm256_add_epi16(u16x16iNume1, u16x16Nume1);

                iCoef += kVecStepX;
            }

            // dst[dstX] = clamp<int>(0, 255, round(nume));
            __m256i u8x32Dst = cvt_roundu16q7_epu8(u16x16Nume0, u16x16Nume1);
            _mm256_storeu_si256((__m256i *)&dst[dstX], u8x32Dst);

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
    IAreaResizerImpl * AreaResizerImpl_new<ArchAVX2FMA>()
    {
        return NULL;
    }

}

#endif
