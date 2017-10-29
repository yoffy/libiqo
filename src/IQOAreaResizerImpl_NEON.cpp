#include "IQOAreaResizerImpl.hpp"


#if defined(IQO_CPU_ARM) && defined(IQO_HAVE_NEON)

#include <cstring>
#include <vector>
#include <arm_neon.h>

#if defined(_OPENMP)
    #include <omp.h>
#endif

#include "math.hpp"
#include "util.hpp"
#include "IQOHWCap.hpp"


namespace {

    //! (uint8_t)min(255, max(0, round(v))) in Q7
    uint8_t cvt_roundu16q7_u8(uint16_t v)
    {
        uint16_t k0_5 = 1 << 6; // 0.5 in Q7
        return uint8_t((v + k0_5) >> 7);
    }


    //! (uint8_t)min(255, max(0, round(v))) in Q7
    uint8x16_t cvt_roundu16q7_u8(uint16x8_t lo, uint16x8_t hi)
    {
        // Round = (uint8_t)round(x / (1<<7))
        uint8x8_t u8x8Round0 = vqrshrn_n_u16(lo, 7);
        uint8x8_t u8x8Round1 = vqrshrn_n_u16(hi, 7);

        return vcombine_u8(u8x8Round0, u8x8Round1);
    }

    int16x8_t gather(const int16_t * src, uint16x8_t indices)
    {
        int16x8_t v = int16x8_t();
        v = vld1q_lane_s16(&src[vgetq_lane_u16(indices, 0)], v, 0);
        v = vld1q_lane_s16(&src[vgetq_lane_u16(indices, 1)], v, 1);
        v = vld1q_lane_s16(&src[vgetq_lane_u16(indices, 2)], v, 2);
        v = vld1q_lane_s16(&src[vgetq_lane_u16(indices, 3)], v, 3);
        v = vld1q_lane_s16(&src[vgetq_lane_u16(indices, 4)], v, 4);
        v = vld1q_lane_s16(&src[vgetq_lane_u16(indices, 5)], v, 5);
        v = vld1q_lane_s16(&src[vgetq_lane_u16(indices, 6)], v, 6);
        v = vld1q_lane_s16(&src[vgetq_lane_u16(indices, 7)], v, 7);
        return v;
    }

}

namespace iqo {

    template<>
    class AreaResizerImpl<ArchNEON> : public IAreaResizerImpl
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
            int16_t * dst
        );

        void resizeYmain(
            ptrdiff_t srcSt, const uint8_t * src,
            int32_t dstW, int16_t * dst,
            int32_t srcOY,
            const int16_t * coefs
        );
        void resizeX(const int16_t * src, uint8_t * dst);
        void resizeXmain(const int16_t * src, uint8_t * dst);

        enum {
            kBiasXBit   = 15,
            kBiasX      = 1 << kBiasXBit,
            kBiasYBit   = 7, // it can be 8, but VQRDMULH is signed
            kBiasY      = 1 << kBiasYBit,

            //! for SIMD
            kVecStepX  = 16, //!< int16x8 x 2
            kVecStepY  = 16, //!< int16x8 x 2
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
        std::vector<int16_t> m_TablesX_;    //!< Area table * m_NumCoordsX (unrolled)
        int16_t * m_TablesX;                //!< aligned
        std::vector<int16_t> m_TablesY;     //!< Area table * m_NumCoordsY
        std::vector<uint16_t> m_IndicesX;
        int32_t m_WorkW;                    //!< width of m_Work
        int16_t * m_Work;
    };

    template<>
    IAreaResizerImpl * AreaResizerImpl_new<ArchNEON>()
    {
        return new AreaResizerImpl<ArchNEON>();
    }


    //! Constructor
    AreaResizerImpl<ArchNEON>::AreaResizerImpl()
    {
        m_Work = NULL;
    }
    // Destructor
    AreaResizerImpl<ArchNEON>::~AreaResizerImpl()
    {
        alignedFree(m_Work);
    }

    // Construct
    void AreaResizerImpl<ArchNEON>::init(
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
        m_TablesX = reinterpret_cast<int16_t *>(addrTablesX);
        size_t tblYSize = size_t(m_NumCoefsY) * m_NumCoordsY;
        m_TablesY.reserve(tblYSize);
        m_TablesY.resize(tblYSize);

        std::vector<int16_t> tablesX(m_NumCoefsX * m_NumCoordsX);
        std::vector<float> coefsX(m_NumCoefsX);
        for ( int32_t dstX = 0; dstX < m_NumCoordsX; ++dstX ) {
            int16_t * table = &tablesX[dstX * m_NumCoefsX];
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
            int16_t * table = &m_TablesY[dstY * m_NumCoefsY];
            float sumCoefs = setAreaTable(rSrcH, rDstH, dstY, m_NumCoefsY, &coefsY[0]);
            adjustCoefs(&coefsY[0], &coefsY[m_NumCoefsY], sumCoefs, kBiasY, &table[0]);
        }

        // allocate workspace
        m_WorkW = alignCeil<int32_t>(m_SrcW, kVecStepY);
        int32_t workSize = m_WorkW * HWCap::getNumberOfProcs();
        m_Work = alignedAlloc<int16_t>(workSize, sizeof(*m_Work) * kVecStepY);

        // calc indices
        int32_t alignedDstW = alignCeil<int32_t>(m_DstW, kVecStepX);
        m_IndicesX.reserve(alignedDstW);
        m_IndicesX.resize(alignedDstW);
        for ( int32_t dstX = 0; dstX < alignedDstW; ++dstX ) {
            //       srcOX = floor(dstX / scale)
            uint16_t srcOX = uint16_t(int64_t(dstX) * rSrcW / rDstW);
            m_IndicesX[dstX] = srcOX;
        }
    }

    void AreaResizerImpl<ArchNEON>::adjustCoefs(
        float * __restrict srcBegin, float * __restrict srcEnd,
        float srcSum,
        uint16_t bias,
        int16_t * __restrict dst)
    {
        const int k1_0 = bias;
        size_t numCoefs = srcEnd - srcBegin;
        int dstSum = 0;

        for ( size_t i = 0; i < numCoefs; ++i ) {
            dst[i] = int16_t(round(srcBegin[i] * bias / srcSum));
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

    void AreaResizerImpl<ArchNEON>::resize(
        size_t srcSt, const uint8_t * src,
        size_t dstSt, uint8_t * __restrict dst
    ) {
        int32_t srcW = m_SrcW;
        int32_t srcH = m_SrcH;
        int32_t dstH = m_DstH;

        if ( srcH == dstH ) {
#pragma omp parallel for
            for ( int32_t y = 0; y < srcH; ++y ) {
                int16_t * work = &m_Work[HWCap::getThreadNumber() * ptrdiff_t(m_WorkW)];
                for ( int32_t x = 0; x < srcW; ++x ) {
                    work[x] = src[srcSt * y + x];
                }
                resizeX(work, &dst[dstSt * y]);
            }
            return;
        }

        const int16_t * tablesY = &m_TablesY[0];

        // main loop
#pragma omp parallel for
        for ( int32_t dstY = 0; dstY < dstH; ++dstY ) {
            int16_t * work = &m_Work[HWCap::getThreadNumber() * ptrdiff_t(m_WorkW)];
            int32_t srcOY = int32_t(int64_t(dstY) * srcH / dstH);
            const int16_t * coefs = &tablesY[dstY % m_NumCoordsY * ptrdiff_t(m_NumCoefsY)];
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
    void AreaResizerImpl<ArchNEON>::resizeYmain(
        ptrdiff_t srcSt, const uint8_t * src,
        int32_t dstW, int16_t * __restrict dst,
        int32_t srcOY,
        const int16_t * coefs
    ) {
        int32_t numCoefsY = m_NumCoefsY;
        int32_t vecLen = alignFloor<int32_t>(dstW, kVecStepY);

#if defined(__GNUC__)
        dst = reinterpret_cast<int16_t *>(__builtin_assume_aligned(dst, sizeof(*dst)*kVecStepY));
#endif

        for ( int32_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            //          nume        = 0;
            int16x8_t   s16x8Nume0  = vdupq_n_s16(0);
            int16x8_t   s16x8Nume1  = vdupq_n_s16(0);

            for ( int16_t i = 0; i < numCoefsY; ++i ) {
                int32_t     srcY        = srcOY + i;

                //          coef        = coefs[i];
                int16x8_t   s16x8Coef   = vdupq_n_s16(coefs[i]);

                //          nume       += src[dstX + srcSt*srcY] * coef;
                uint8x16_t  u8x16Src    = vld1q_u8(&src[dstX + srcSt*srcY]);
                int16x8_t   s16x8Src0   = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u8x16Src)));
                int16x8_t   s16x8Src1   = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u8x16Src)));
                s16x8Nume0 = vmlaq_s16(s16x8Nume0, s16x8Src0, s16x8Coef);
                s16x8Nume1 = vmlaq_s16(s16x8Nume1, s16x8Src1, s16x8Coef);
            }

            // dst[dstX] = nume;
            vst1q_s16(&dst[dstX + 0], s16x8Nume0);
            vst1q_s16(&dst[dstX + 8], s16x8Nume1);
        }

        for ( int32_t dstX = vecLen; dstX < dstW; dstX++ ) {
            int nume = 0;

            for ( int16_t i = 0; i < numCoefsY; ++i ) {
                int32_t srcY = srcOY + i;
                int16_t coef = coefs[i];
                nume += src[dstX + srcSt * srcY] * coef;
            }

            dst[dstX] = int16_t(nume);
        }
    }

    void AreaResizerImpl<ArchNEON>::resizeX(const int16_t * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            int32_t dstW = m_DstW;
            for ( int32_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = cvt_roundu16q7_u8(src[dstX]);
            }
            return;
        }

        resizeXmain(src, dst);
    }

    //! resize horizontal (main loop)
    //!
    //! @param src    A row of source
    //! @param dst    A row of destination
    void AreaResizerImpl<ArchNEON>::resizeXmain(const int16_t * src, uint8_t * __restrict dst)
    {
        const int16_t * coefs = &m_TablesX[0];
        const uint16_t * indices = &m_IndicesX[0];
        ptrdiff_t tableWidth = ptrdiff_t(m_TablesXWidth);
        ptrdiff_t tableSize = tableWidth * m_NumUnrolledCoordsX;
        int32_t numCoefsX = m_NumCoefsX;
        int32_t dstW = m_DstW;
        int32_t vecLen = alignFloor<int32_t>(dstW, kVecStepX);

        ptrdiff_t iCoef = 0;
        for ( int32_t dstX = 0; dstX < vecLen; dstX += kVecStepX ) {
            //          nume        = 0;
            int16x8_t   s16x8Nume0  = vdupq_n_s16(0);
            int16x8_t   s16x8Nume1  = vdupq_n_s16(0);
            //          srcOX       = floor(dstX / scale);
            uint16x8_t  u16x8SrcOX0 = vld1q_u16(&indices[dstX + 0]);
            uint16x8_t  u16x8SrcOX1 = vld1q_u16(&indices[dstX + 8]);

            for ( int16_t i = 0; i < numCoefsX; ++i ) {
                //          srcX        = srcOX + i;
                uint16x8_t  u16x8Offset = vdupq_n_u16(i);
                uint16x8_t  u16x8SrcX0  = vaddq_u16(u16x8SrcOX0, u16x8Offset);
                uint16x8_t  u16x8SrcX1  = vaddq_u16(u16x8SrcOX1, u16x8Offset);

                //          nume       += src[srcX] * coefs[iCoef];
                int16x8_t   s16x8Src0   = gather(src, u16x8SrcX0);
                int16x8_t   s16x8Src1   = gather(src, u16x8SrcX1);
                int16x8_t   s16x8Coefs0 = vld1q_s16(&coefs[iCoef + 0]);
                int16x8_t   s16x8Coefs1 = vld1q_s16(&coefs[iCoef + 8]);
                //                        (src*kBiasY * coef*kBiasX) / kBiasX
                int16x8_t   s16x8iNume0 = vqrdmulhq_s16(s16x8Src0, s16x8Coefs0); // 16bit of 33bit???
                int16x8_t   s16x8iNume1 = vqrdmulhq_s16(s16x8Src1, s16x8Coefs1);
                s16x8Nume0 = vaddq_s16(s16x8Nume0, s16x8iNume0);
                s16x8Nume1 = vaddq_s16(s16x8Nume1, s16x8iNume1);

                iCoef += kVecStepX;
            }

            // dst[dstX] = clamp<int>(0, 255, round(nume));
            uint16x8_t  u16x8Dst0   = vreinterpretq_u16_s16(s16x8Nume0);
            uint16x8_t  u16x8Dst1   = vreinterpretq_u16_s16(s16x8Nume1);
            uint8x16_t  u8x16Dst    = cvt_roundu16q7_u8(u16x8Dst0, u16x8Dst1);
            vst1q_u8(&dst[dstX], u8x16Dst);

            // iCoef = dstX % tableSize;
            if ( iCoef == tableSize ) {
                iCoef = 0;
            }
        }

        for ( int32_t dstX = vecLen; dstX < dstW; ++dstX ) {
            int32_t srcOX = indices[dstX];
            int     sum = 0;

            // calc index of coefs from unrolled table
            iCoef = (dstX % kVecStepX) + (dstX / kVecStepX % m_NumUnrolledCoordsX * tableWidth);

            for ( int16_t i = 0; i < numCoefsX; ++i ) {
                int32_t srcX = srcOX + i;
                sum   += (src[srcX] * coefs[iCoef]) / kBiasX;
                iCoef += kVecStepX;
            }

            dst[dstX] = cvt_roundu16q7_u8(uint16_t(sum));
        }
    }

}

#else

namespace iqo {

    template<>
    IAreaResizerImpl * AreaResizerImpl_new<ArchNEON>()
    {
        return NULL;
    }

}

#endif
