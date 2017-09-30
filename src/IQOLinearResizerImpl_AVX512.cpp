#include "IQOLinearResizerImpl.hpp"


#if defined(IQO_CPU_X86) && defined(IQO_HAVE_AVX512)

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

    uint8_t cvt_roundss_su8(float v)
    {
        __m128  f32x1V     = _mm_set_ss(v);
        __m128  f32x1Round = _mm_round_ss(f32x1V, f32x1V, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        int32_t s32x1Round = _mm_cvtss_si32(f32x1Round);
        return uint8_t(iqo::clamp(0, 255, s32x1Round));
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
        __m256i u8x32  = _mm256_permutevar8x32_epi32(u8x32P, u32x8Table);
        return u8x32;
    }

}

namespace iqo {

    template<>
    class LinearResizerImpl<ArchAVX512> : public ILinearResizerImpl
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

        //! get index of m_TablesX from coordinate of destination
        ptrdiff_t getCoefXIndex(int32_t dstX);

        enum {
            m_NumCoefsX = 2,
            m_NumCoefsY = 2,

            //! for SIMD
            kVecStepY  = 32, //!< __m512 x 2
            kVecStepX  = 32, //!< __m512 x 2
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
    bool LinearResizerImpl_hasFeature<ArchAVX512>()
    {
        HWCap cap;
        return cap.hasAVX512F()
            && cap.hasAVX512VL()
            && cap.hasAVX512BW()
            && cap.hasAVX512DQ()
            && cap.hasAVX512CD();
    }

    template<>
    ILinearResizerImpl * LinearResizerImpl_new<ArchAVX512>()
    {
        return new LinearResizerImpl<ArchAVX512>();
    }


    // Constructor
    void LinearResizerImpl<ArchAVX512>::init(
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

    void LinearResizerImpl<ArchAVX512>::resize(
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
        int32_t mainBegin0 = convertCoordinate(srcH, dstH, 0);
        int32_t mainBegin  = clamp<int32_t>(0, dstH, mainBegin0);
        int32_t mainEnd    = clamp<int32_t>(0, dstH, dstH - mainBegin);

        // border pixels
#pragma omp parallel for
        for ( int32_t dstY = 0; dstY < mainBegin; ++dstY ) {
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
        for ( int32_t dstY = mainBegin; dstY < mainEnd; ++dstY ) {
            float * work = &m_Work[HWCap::getThreadNumber() * int32_t(srcW)];
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
        for ( int32_t dstY = mainEnd; dstY < m_DstH; ++dstY ) {
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
    void LinearResizerImpl<ArchAVX512>::resizeYborder(
        ptrdiff_t srcSt, const uint8_t * src,
        int32_t dstW, float * __restrict dst,
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
    void LinearResizerImpl<ArchAVX512>::resizeYmain(
        ptrdiff_t srcSt, const uint8_t * src,
        int32_t dstW, float * __restrict dst,
        int32_t srcOY,
        const float * coefs
    ) {
        int32_t numCoefsY = m_NumCoefsY;
        int32_t vecLen = alignFloor<int32_t>(dstW, kVecStepY);

        for ( int32_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            //     nume         = 0;
            __m512 f32x16Nume0  = _mm512_setzero_ps();
            __m512 f32x16Nume1  = _mm512_setzero_ps();

            for ( int32_t i = 0; i < numCoefsY; ++i ) {
                int32_t srcY = srcOY + i;

                //      coef        = coefs[i];
                __m512  f32x16Coef  = _mm512_set1_ps(coefs[i]);

                //      nume       += src[dstX + srcSt*srcY] * coef;
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

    void LinearResizerImpl<ArchAVX512>::resizeX(const float * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            int32_t dstW = m_DstW;
            for ( int32_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = cvt_roundss_su8(src[dstX]);
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
    void LinearResizerImpl<ArchAVX512>::resizeXborder(
        const float * src, uint8_t * __restrict dst,
        int32_t mainBegin, int32_t mainEnd,
        int32_t begin, int32_t end
    ) {
        const float * coefs = &m_TablesX[0];
        const int32_t * indices = &m_IndicesX[0];
        int32_t numCoefsX = m_NumCoefsX;
        int32_t srcW = m_SrcW;

        for ( int32_t dstX = begin; dstX < end; ++dstX ) {
            int32_t srcOX = indices[dstX];
            float   sum   = 0;

            if ( dstX < mainBegin ) {
                dst[dstX] = cvt_roundss_su8(src[0]);
                continue;
            }
            if ( mainEnd <= dstX ) {
                dst[dstX] = cvt_roundss_su8(src[srcW - 1]);
                continue;
            }

            ptrdiff_t iCoef = getCoefXIndex(dstX);
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
    void LinearResizerImpl<ArchAVX512>::resizeXmain(
        const float * src, uint8_t * __restrict dst,
        int32_t begin, int32_t end
    ) {
        const float * coefs = &m_TablesX[0];
        const int32_t * indices = &m_IndicesX[0];
        ptrdiff_t tableWidth = ptrdiff_t(m_TablesXWidth);
        ptrdiff_t tableSize = tableWidth * m_NumUnrolledCoordsX;
        int32_t numCoefsX = m_NumCoefsX;

        ptrdiff_t iCoef = getCoefXIndex(begin);
        for ( int32_t dstX = begin; dstX < end; dstX += kVecStepX ) {
            //      nume            = 0;
            __m512  f32x16Nume0     = _mm512_setzero_ps();
            __m512  f32x16Nume1     = _mm512_setzero_ps();
            //      srcOX           = int32_t(floor((dstX+0.5) / scale - 0.5));
            __m512i s32x16SrcOX0    = _mm512_loadu_si512((const __m512i*)&indices[dstX +  0]);
            __m512i s32x16SrcOX1    = _mm512_loadu_si512((const __m512i*)&indices[dstX + 16]);

            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                //      srcX            = srcOX + i;
                __m512i s32x16Offset    = _mm512_set1_epi32(i);
                __m512i s32x16SrcX0     = _mm512_add_epi32(s32x16SrcOX0, s32x16Offset);
                __m512i s32x16SrcX1     = _mm512_add_epi32(s32x16SrcOX1, s32x16Offset);

                //      nume           += src[srcX] * coefs[iCoef];
                __m512  s32x16Src0      = _mm512_i32gather_ps(s32x16SrcX0, src, sizeof(float));
                __m512  s32x16Src1      = _mm512_i32gather_ps(s32x16SrcX1, src, sizeof(float));
                __m512  f32x16Coefs0    = _mm512_load_ps(&coefs[iCoef +  0]);
                __m512  f32x16Coefs1    = _mm512_load_ps(&coefs[iCoef + 16]);
                f32x16Nume0 = _mm512_fmadd_ps(s32x16Src0, f32x16Coefs0, f32x16Nume0);
                f32x16Nume1 = _mm512_fmadd_ps(s32x16Src1, f32x16Coefs1, f32x16Nume1);

                iCoef += kVecStepX;
            }

            // dst[dstX] = clamp<int>(0, 255, round(nume));
            __m256i u8x32Dst = cvt_roundps_epu8(f32x16Nume0, f32x16Nume1);
            _mm256_storeu_si256((__m256i*)&dst[dstX], u8x32Dst);

            // iCoef = dstX % tableSize;
            if ( iCoef == tableSize ) {
                iCoef = 0;
            }
        }
    }

    //! get index of m_TablesX from coordinate of destination
    ptrdiff_t LinearResizerImpl<ArchAVX512>::getCoefXIndex(int32_t dstX)
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
    bool LinearResizerImpl_hasFeature<ArchAVX512>()
    {
        return false;
    }

    template<>
    ILinearResizerImpl * LinearResizerImpl_new<ArchAVX512>()
    {
        return NULL;
    }

}

#endif
