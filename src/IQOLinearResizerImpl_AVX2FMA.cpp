#include "IQOLinearResizerImpl.hpp"


#if defined(IQO_CPU_X86) && defined(IQO_HAVE_AVX2FMA)

#include <cstring>
#include <vector>
#include <immintrin.h>

#include "math.hpp"
#include "IQOHWCap.hpp"


namespace {

    //! convert pixel scale of coordinate
    //!
    //! @see iqo::setLinearTable
    int32_t convertCoordinate(int32_t fromX, int32_t fromLen, int32_t toLen)
    {
        // When magnification (fromLen < toLen), toX is grater than 0.
        // In this case, no calculate border ceil(toX) pixels.
        //
        // When reducing (fromLen > toLen), toX between -0.5 to 0.0.
        // In this case, no calculate border 1 pixel by ceil(fabs(toX))
        double  toX = (0.5 + fromX) * toLen / fromLen - 0.5;
        return int32_t(std::ceil(std::fabs(toX)));
    }

    //! (uint8_t)min(255, max(0, round(v))) in Q8.8
    uint8_t cvt_roundu16q8_su8(uint16_t v)
    {
        uint16_t k0_5 = 1 << 7; // 0.5 in Q8.8
        return uint8_t((v + k0_5) >> 8);
    }

    //! (uint8_t)min(255, max(0, round(v))) in Q8.8
    __m256i cvt_roundu16q8_epu8(__m256i lo, __m256i hi)
    {
        __m256i u16x16k0_5 = _mm256_set1_epi16(1 << 7); // 0.5 in Q8.8

        // v = (x + k0_5) >> 8
        __m256i u16x16v0 = _mm256_srli_epi16(_mm256_add_epi16(lo, u16x16k0_5), 8);
        __m256i u16x16v1 = _mm256_srli_epi16(_mm256_add_epi16(hi, u16x16k0_5), 8);

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

    //! returns { v.i64[I0], v.i64[I1], v.i64[I2], v.i64[I3] };
    template<int I0, int I1, int I2, int I3>
    __m256i permute4xi64(__m256i v)
    {
        return _mm256_permute4x64_epi64(v, _MM_SHUFFLE(I3, I2, I1, I0));
    }

    //! (uint8_t)min(255, max(0, round(v))) in Q8.8
    __m256i packus_epi16(__m256i u16x16v0, __m256i u16x16v1)
    {
        // Perm = (uint8_t)v
        __m256i u8x32Perm = _mm256_packus_epi16(u16x16v0, u16x16v1);
        return permute4xi64<0, 2, 1, 3>(u8x32Perm);
    }

    //! (uint16_t)min(255, max(0, round(v))) in Q16.16
    __m256i cvt_roundu32q16_epu16(__m256i v0, __m256i v1)
    {
        __m256i u32x8k0_5 = _mm256_set1_epi32(1 << 15); // 0.5 in Q16.16

        // v = (x + k0_5) >> 16;
        __m256i u32x8v0 = _mm256_srli_epi32(_mm256_add_epi32(v0, u32x8k0_5), 16);
        __m256i u32x8v1 = _mm256_srli_epi32(_mm256_add_epi32(v1, u32x8k0_5), 16);

        // Perm = (uint16_t)v;
        __m256i u16x16Perm = _mm256_packus_epi32(u32x8v0, u32x8v1);
        return permute4xi64<0, 2, 1, 3>(u16x16Perm);
    }

    //! (uint8_t)min(255, max(0, round(v))) in Q16.16
    __m256i cvt_roundu32q16_epu8(__m256i v0, __m256i v1, __m256i v2, __m256i v3)
    {
        __m256i u16x16v0 = cvt_roundu32q16_epu16(v0, v1);
        __m256i u16x16v1 = cvt_roundu32q16_epu16(v2, v3);
        return packus_epi16(u16x16v0, u16x16v1);
    }

    //! returns { src[indices[i]], src[indices[i] + 1] }
    __m256i gatheru16x2(const uint16_t * src, const uint16_t * indices)
    {
        const __m128i k0 = _mm_setzero_si128();

        // i0 i2 i4 i6 i8 i10 i12 i14
        __m128i u16x8indices   = _mm_loadu_si128((const __m128i *)indices);
        // i0 i2 i4 i6
        __m128i s32x4indices0  = _mm_unpacklo_epi16(u16x8indices, k0);
        __m128i s32x4indices1  = _mm_unpackhi_epi16(u16x8indices, k0);
        // i0 i2 i4 i6 i8 i10 i12 i14
        __m256i s32x8indices   = _mm256_set_m128i(s32x4indices1, s32x4indices0);
        // x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15
        return _mm256_i32gather_epi32((const int *)src, s32x8indices, sizeof(u16));
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
        //! returns { coef0, coef1, .., coef15 } from odd coeficients
        //!
        //! @code
        //! from: { coef1, coef3, .., coef15 } ( 8 elements)
        //!   to: { coef0, coef1, .., coef15 } (16 elements)
        //! @endcode
        static __m256i loadCoefsX(const uint16_t * coefs);

        //! dst[i] = src[i] * kBias / srcSum (src will be broken)
        void adjustCoefs(
            float * srcBegin, float * srcEnd,
            uint16_t bias,
            uint16_t * dst
        );

        void resizeYborder(
            ptrdiff_t srcSt, const uint8_t * src,
            int32_t dstW, uint16_t * dst,
            int32_t srcOY
        );
        void resizeYmain(
            ptrdiff_t srcSt, const uint8_t * src,
            int32_t dstW, uint16_t * dst,
            int32_t srcOY,
            const uint16_t * coefs
        );
        void resizeX(const uint16_t * src, uint8_t * dst);
        void resizeXborder(
            const uint16_t * src, uint8_t * dst,
            int32_t mainBegin, int32_t mainEnd,
            int32_t begin, int32_t end
        );
        void resizeXmain(
            const uint16_t * src, uint8_t * dst,
            int32_t begin, int32_t end
        );

        //! get index of m_TablesX from coordinate of destination
        ptrdiff_t getCoefXIndex(int32_t dstX);

        enum {
            m_NumCoefsX = 2,
            m_NumCoefsY = 2,

            kBiasXBit   = 8,
            kBiasX      = 1 << kBiasXBit,
            kBiasYBit   = 8,
            kBiasY      = 1 << kBiasYBit,

            //! for SIMD
            kVecStepX   = 32, //!< uint16x16 x 2
            kVecStepY   = 32, //!< uint16x16 x 2
        };
        int32_t m_SrcW;
        int32_t m_SrcH;
        int32_t m_DstW;
        int32_t m_DstH;
        int32_t m_NumCoordsX;
        int32_t m_NumUnrolledCoordsX;
        int32_t m_TablesXWidth;
        int32_t m_NumCoordsY;
        std::vector<uint16_t> m_TablesX_;  //!< Linear table * m_NumCoordsX (unrolled)
        uint16_t * m_TablesX;              //!< aligned
        std::vector<uint16_t> m_TablesY;   //!< Linear table * m_NumCoordsY
        std::vector<uint16_t> m_IndicesX;
        std::vector<uint16_t> m_Work;
    };

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
        m_TablesX = reinterpret_cast<uint16_t *>(addrTablesX);
        size_t tblYSize = size_t(m_NumCoefsY) * m_NumCoordsY;
        m_TablesY.reserve(tblYSize);
        m_TablesY.resize(tblYSize);

        // X coefs
        std::vector<float> fTablesX(rDstW);
        std::<uint16_t> tablesX(rDstW);
        setLinearTable(rSrcW, rDstW, &fTablesX[0]);
        adjustCoefs(&fTablesX[0], &fTablesX[rDstW], kBiasX, &tablesX[0]);

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
        std::vector<float> fTablesY(tblYSize);
        setLinearTable(rSrcH, rDstH, &fTablesY[0]);
        adjustCoefs(&m_TablesY[0], &fTablesY[m_NumCoefsY], kBiasY, &m_TablesY[0]);

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
            //uint16_t srcOX = floor((dstX+0.5) * srcW/dstW - 0.5);
            uint16_t srcOX = uint16_t(*iSrcOX++);
            m_IndicesX[dstX] = srcOX;
        }
    }

    //! returns { coef0, coef1, .., coef15 } from odd coeficients
    //!
    //! @code
    //! from: { coef1, coef3, .., coef15 } ( 8 elements)
    //!   to: { coef0, coef1, .., coef15 } (16 elements)
    //! @endcode
    __m256i LinearResizerImpl<ArchAVX2FMA>::loadCoefsX(const uint16_t * coefs)
    {
        const __m128i u16x8kBias = _mm_set1_epi16(kBiasX);

        // c1 c3 c5 c7 c9 c11 c13 c15
        __m128i u16x8coefOd = _mm_loadu_si128((const __m128i *)coefs);
        // c0 c2 c4 c6 c8 c10 c12 c14
        __m128i u16x8coefEv = _mm_sub_epi16(u16x8kBias, u16x8coefOd);
        // c0 c1 c2 c3 c4 c5 c6 c7
        __m128i u16x8coef0 = _mm_unpacklo_epi16(u16x8coefEv, u16x8coefOd);
        __m128i u16x8coef1 = _mm_unpackhi_epi16(u16x8coefEv, u16x8coefOd);
        // c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
        return _mm256_set_m128i(u16x8coef1, u16x8coef0);
    }

    void LinearResizerImpl<ArchAVX2FMA>::adjustCoefs(
        float * __restrict srcBegin, float * __restrict srcEnd,
        uint16_t bias,
        uint16_t * __restrict dst)
    {
        const int k1_0 = bias;
        size_t numCoefs = srcEnd - srcBegin;

        for ( size_t i = 0; i < numCoefs; i += 2 ) {
            dst[i + 0] = uint16_t(round(srcBegin[i] * bias));
            dst[i + 1] = k1_0 - dst[i + 0];
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
                uint16_t * work = &m_Work[HWCap::getThreadNumber() * ptrdiff_t(srcW)];
                for ( int32_t x = 0; x < srcW; ++x ) {
                    work[x] = src[srcSt * y + x];
                }
                resizeX(work, &dst[dstSt * y]);
            }
            return;
        }

        const uint16_t * tablesY = &m_TablesY[0];
        int32_t mainBegin0 = convertCoordinate(srcH, dstH, 0);
        int32_t mainBegin  = clamp<int32_t>(0, dstH, mainBegin0);
        int32_t mainEnd    = clamp<int32_t>(0, dstH, dstH - mainBegin);

        // border pixels
#pragma omp parallel for
        for ( int32_t dstY = 0; dstY < mainBegin; ++dstY ) {
            uint16_t * work = &m_Work[HWCap::getThreadNumber() * ptrdiff_t(srcW)];
            int32_t srcOY = 0;
            resizeYborder(
                srcSt, &src[0],
                srcW, work,
                srcOY);
            resizeX(work, &dst[dstSt * dstY]);
        }

        // main loop
#pragma omp parallel for
        for ( int32_t dstY = mainBegin; dstY < mainEnd; ++dstY ) {
            uint16_t * work = &m_Work[HWCap::getThreadNumber() * ptrdiff_t(srcW)];
            int32_t srcOY = int32_t(floor((dstY+0.5) * srcH/dstH - 0.5));
            const uint16_t * coefs = &tablesY[dstY % m_NumCoordsY * ptrdiff_t(m_NumCoefsY)];
            resizeYmain(
                srcSt, &src[0],
                srcW, work,
                srcOY,
                coefs);
            resizeX(work, &dst[dstSt * dstY]);
        }

        // border pixels
#pragma omp parallel for
        for ( int32_t dstY = mainEnd; dstY < m_DstH; ++dstY ) {
            uint16_t * work = &m_Work[HWCap::getThreadNumber() * ptrdiff_t(srcW)];
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
        int32_t dstW, uint16_t * __restrict dst,
        int32_t srcOY
    ) {
        for ( int32_t dstX = 0; dstX < dstW; ++dstX ) {
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
        int32_t dstW, uint16_t * __restrict dst,
        int32_t srcOY,
        const uint16_t * coefs
    ) {
        int32_t numCoefsY = m_NumCoefsY;
        int32_t vecLen = alignFloor<int32_t>(dstW, kVecStepY);

        for ( int32_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            //      nume            = 0;
            __m256i u16x16Nume0    = _mm256_setzero_si256(); // permuted by unpack
            __m256i u16x16Nume1    = _mm256_setzero_si256();

            for ( int32_t i = 0; i < numCoefsY; ++i ) {
                int32_t srcY = srcOY + i;

                //      coef        = coefs[i];
                __m256i u16x16Coef   = _mm256_set1_epi16(coefs[i]);

                //      nume       += src[dstX + srcSt*srcY] * coef;
                __m256i u8x32Src    = _mm256_loadu_si256((const __m256i *)&src[dstX + srcSt*srcY]);
                _mm_prefetch((const char *)&src[dstX + srcSt*(srcY + numCoefsY)], _MM_HINT_T2);
                __m256i u16x16Src0   = _mm256_unpacklo_epi8(u8x32Src, _mm256_setzero_si256());
                __m256i u16x16Src1   = _mm256_unpackhi_epi8(u8x32Src, _mm256_setzero_si256());
                __m256i u16x16iNume0 = _mm256_mullo_epi16(u16x16Src0, u16x16Coef);
                __m256i u16x16iNume1 = _mm256_mullo_epi16(u16x16Src1, u16x16Coef);
                u16x16Nume0 = _mm256_add_epi16(u16x16Nume0, u16x16iNume0);
                u16x16Nume1 = _mm256_add_epi16(u16x16Nume1, u16x16iNume1);
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

    void LinearResizerImpl<ArchAVX2FMA>::resizeX(const uint16_t * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            int32_t dstW = m_DstW;
            for ( int32_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = cvt_roundu16q8_su8(src[dstX]);
            }
            return;
        }

        int32_t dstW        = m_DstW;
        int32_t mainBegin0  = convertCoordinate(m_SrcW, dstW, 0);
        int32_t mainBegin   = clamp<int32_t>(0, dstW, mainBegin0);
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
        const uint16_t * src, uint8_t * __restrict dst,
        int32_t mainBegin, int32_t mainEnd,
        int32_t begin, int32_t end
    ) {
        const uint16_t * coefs = &m_TablesX[0];
        const uint16_t * indices = &m_IndicesX[0];
        int32_t numCoefsX = m_NumCoefsX;
        int32_t srcW = m_SrcW;

        for ( int32_t dstX = begin; dstX < end; ++dstX ) {

            if ( dstX < mainBegin ) {
                dst[dstX] = cvt_roundu16q8_su8(src[0]);
                continue;
            }
            if ( mainEnd <= dstX ) {
                dst[dstX] = cvt_roundu16q8_su8(src[srcW - 1]);
                continue;
            }

            uint16_t srcX   = indices[dstX];
            ptrdiff_t iCoef = getCoefXIndex(dstX);
            uint16_t coef0  = coefs[iCoef + 0];
            uint16_t coef1  = coefs[iCoef + 1];
            uint16_t src0   = src[srcX + 0];
            uint16_t src1   = src[srcX + 0];
            uint16_t sum    = uint16_t((src0*coef0 + src1*coef1) >> 8);

            dst[dstX] = cvt_roundu16q8_su8(sum);
        }
    }

    //! resize horizontal (main loop)
    //!
    //! @param src    A row of source
    //! @param dst    A row of destination
    //! @param begin  Position of a first pixel
    //! @param end    Position of next of a last pixel
    void LinearResizerImpl<ArchAVX2FMA>::resizeXmain(
        const uint16_t * src, uint8_t * __restrict dst,
        int32_t begin, int32_t end
    ) {
        const uint16_t * coefs = &m_TablesX[0];
        const uint16_t * indices = &m_IndicesX[0];
        ptrdiff_t tableWidth = ptrdiff_t(m_TablesXWidth);
        ptrdiff_t tableSize = tableWidth * m_NumUnrolledCoordsX;
        int32_t numCoefsX = m_NumCoefsX;

        ptrdiff_t iCoef = getCoefXIndex(begin);
        for ( int32_t dstX = begin; dstX < end; dstX += kVecStepX ) {
            //      nume           += src[srcX+0]*coefs[iCoef+0] + src[srcX+1]*coefs[iCoef+1];
            __m256i u16x16Src0      = gatheru16x2(src, &indices[dstX +  0]);
            __m256i u16x16Src1      = gatheru16x2(src, &indices[dstX + 16]);
            __m256i u16x16Src2      = gatheru16x2(src, &indices[dstX + 32]);
            __m256i u16x16Src3      = gatheru16x2(src, &indices[dstX + 48]);
            __m256i u16x16Coefs0    = _mm256_load_si256((const __m256i *)&coefs[iCoef +  0]);
            __m256i u16x16Coefs1    = _mm256_load_si256((const __m256i *)&coefs[iCoef + 16]);
            __m256i u16x16Coefs2    = _mm256_load_si256((const __m256i *)&coefs[iCoef + 32]);
            __m256i u16x16Coefs3    = _mm256_load_si256((const __m256i *)&coefs[iCoef + 48]);
            __m256i u32x8Nume0      = _mm256_madd_epi16(u16x16Src0, u16x16Coefs0);
            __m256i u32x8Nume1      = _mm256_madd_epi16(u16x16Src1, u16x16Coefs1);
            __m256i u32x8Nume2      = _mm256_madd_epi16(u16x16Src2, u16x16Coefs2);
            __m256i u32x8Nume3      = _mm256_madd_epi16(u16x16Src3, u16x16Coefs3);

            //      dst             = clamp<int>(0, 255, round(uint16_t(nume)));
            __m256i u8x32Dst        = cvt_roundu32q16_epu8(
                u32x8Nume0, u32x8Nume1, u32x8Nume2, u32x8Nume3
            );
            _mm256_storeu_si256((__m256i*)&dst[dstX], u8x32Dst);

#warning todo
            iCoef += kVecStepX * 2;

            // iCoef = dstX % tableSize;
            if ( iCoef == tableSize ) {
                iCoef = 0;
            }
        }
    }

    //! get index of m_TablesX from coordinate of destination
    ptrdiff_t LinearResizerImpl<ArchAVX2FMA>::getCoefXIndex(int32_t dstX)
    {
        //      srcX: ABCA--
        //            BCAB |- m_NumUnrolledCoordsX
        //            CABC--
        //
        // m_TablesX:
        //                dstX % kVecStepX
        //                     |
        //                  --------
        //                  |      |
        //                --A0B0C0A0 .. A3B3C3A3
        // dstX/kVecStepX-| B0C0A0B0 .. B3C3A3B3
        //                --C0A0B0C0 .. C3A3B3C3
        //                  |                  |
        //                  --------------------
        //                            |
        //                     m_TablesXWidth
        //
        //

        ptrdiff_t tableWidth = ptrdiff_t(m_TablesXWidth);
        return (dstX % kVecStepX) + (dstX / kVecStepX % m_NumUnrolledCoordsX * tableWidth);
    }

}

#else

namespace iqo {

    template<>
    ILinearResizerImpl * LinearResizerImpl_new<ArchAVX2FMA>()
    {
        return NULL;
    }

}

#endif
