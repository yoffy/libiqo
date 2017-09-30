#include "IQOLinearResizerImpl.hpp"


#if defined(IQO_CPU_ARM) && defined(IQO_HAVE_NEON)

#include <cstring>
#include <vector>
#include <arm_neon.h>

#if defined(_OPENMP)
    #include <omp.h>
#endif

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

    float32x4_t gather(const float * src, int32x4_t indices)
    {
        float32x4_t v = float32x4_t();
        v = vld1q_lane_f32(&src[vgetq_lane_s32(indices, 0)], v, 0);
        v = vld1q_lane_f32(&src[vgetq_lane_s32(indices, 1)], v, 1);
        v = vld1q_lane_f32(&src[vgetq_lane_s32(indices, 2)], v, 2);
        v = vld1q_lane_f32(&src[vgetq_lane_s32(indices, 3)], v, 3);
        return v;
    }

    float round_f32(float v)
    {
        // http://developer.apple.com/legacy/mac/library/documentation/Performance/Conceptual/Accelerate_sse_migration/migration_sse_translation/migration_sse_translation.html#//apple_ref/doc/uid/TP40002729-CH248-279676
        float32x2_t v1_5p23f = { 0x1.5p23f, 0x1.5p23f };
        float32x2_t vSrc = float32x2_t();
        vSrc = vset_lane_f32(v, vSrc, 0);
        float32x2_t vRound = vsub_f32(vadd_f32(vSrc, v1_5p23f), v1_5p23f);
        return vget_lane_f32(vRound, 0);
    }

    float32x4_t round_f32(float32x4_t v)
    {
        // http://developer.apple.com/legacy/mac/library/documentation/Performance/Conceptual/Accelerate_sse_migration/migration_sse_translation/migration_sse_translation.html#//apple_ref/doc/uid/TP40002729-CH248-279676
        float32x4_t v1_5p23f = { 0x1.5p23f, 0x1.5p23f, 0x1.5p23f, 0x1.5p23f };
        return vsubq_f32(vaddq_f32(v, v1_5p23f), v1_5p23f);
    }

    uint8_t cvt_roundf32_u8(float v)
    {
        return uint8_t(iqo::clamp(0.0f, 255.0f, round_f32(v)));
    }

    uint8x8_t cvt_roundf32_u8(float32x4_t lo, float32x4_t hi)
    {
        float32x4_t f32x4Round0 = round_f32(lo);
        float32x4_t f32x4Round1 = round_f32(hi);
        int32x4_t   s32x4Round0 = vcvtq_s32_f32(f32x4Round0);
        int32x4_t   s32x4Round1 = vcvtq_s32_f32(f32x4Round1);
        uint16x4_t  u16x4Round0 = vqmovun_s32(s32x4Round0);
        uint16x4_t  u16x4Round1 = vqmovun_s32(s32x4Round1);
        uint16x8_t  u16x8Round  = vcombine_u16(u16x4Round0, u16x4Round1);
        uint8x8_t   u8x8Round   = vqmovn_u16(u16x8Round);
        return u8x8Round;
    }

}

namespace iqo {

    template<>
    class LinearResizerImpl<ArchNEON> : public ILinearResizerImpl
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
            kVecStepX  =  8, //!< float32x4_t x 2
            kVecStepY  = 16, //!< float32x4_t x 4
        };
        int32_t m_SrcW;
        int32_t m_SrcH;
        int32_t m_DstW;
        int32_t m_DstH;
        int32_t m_NumCoordsX;
        int32_t m_NumUnrolledCoordsX;
        int32_t m_TablesXWidth;
        int32_t m_NumCoordsY;
        std::vector<float> m_TablesX_;  //!< m_TablesXWidth * m_NumCoordsX (unrolled)
        float * m_TablesX;              //!< aligned
        std::vector<float> m_TablesY;   //!< Linear table * m_NumCoordsY
        std::vector<int32_t> m_IndicesX;
        std::vector<float> m_Work;
    };

    template<>
    bool LinearResizerImpl_hasFeature<ArchNEON>()
    {
        HWCap cap;
        return cap.hasNEON();
    }

    template<>
    ILinearResizerImpl * LinearResizerImpl_new<ArchNEON>()
    {
        return new LinearResizerImpl<ArchNEON>();
    }


    // Constructor
    void LinearResizerImpl<ArchNEON>::init(
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

    void LinearResizerImpl<ArchNEON>::resize(
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
    void LinearResizerImpl<ArchNEON>::resizeYborder(
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
    void LinearResizerImpl<ArchNEON>::resizeYmain(
        ptrdiff_t srcSt, const uint8_t * src,
        int32_t dstW, float * __restrict dst,
        int32_t srcOY,
        const float * coefs
    ) {
        int32_t vecLen = alignFloor<int32_t>(dstW, kVecStepY);
        int32_t numCoefsY = m_NumCoefsY;

        for ( int32_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            //          nume       = 0;
            float32x4_t f32x4Nume0 = vdupq_n_f32(0);
            float32x4_t f32x4Nume1 = vdupq_n_f32(0);
            float32x4_t f32x4Nume2 = vdupq_n_f32(0);
            float32x4_t f32x4Nume3 = vdupq_n_f32(0);

            for ( int32_t i = 0; i < numCoefsY; ++i ) {
                int32_t     srcY        = srcOY + i;

                //          coef        = coefs[i];
                float32x4_t f32x4Coef   = vdupq_n_f32(coefs[i]);

                //          nume       += src[dstX + srcSt*srcY] * coef;
                uint8x16_t  u8x16Src    = vld1q_u8(&src[dstX + srcSt*srcY]);
                uint16x8_t  u16x8Src0   = vmovl_u8(vget_low_u8(u8x16Src));
                uint16x8_t  u16x8Src1   = vmovl_u8(vget_high_u8(u8x16Src));
                uint32x4_t  u32x4Src0   = vmovl_u16(vget_low_u16(u16x8Src0));
                uint32x4_t  u32x4Src1   = vmovl_u16(vget_high_u16(u16x8Src0));
                uint32x4_t  u32x4Src2   = vmovl_u16(vget_low_u16(u16x8Src1));
                uint32x4_t  u32x4Src3   = vmovl_u16(vget_high_u16(u16x8Src1));
                float32x4_t f32x4Src0   = vcvtq_f32_u32(u32x4Src0);
                float32x4_t f32x4Src1   = vcvtq_f32_u32(u32x4Src1);
                float32x4_t f32x4Src2   = vcvtq_f32_u32(u32x4Src2);
                float32x4_t f32x4Src3   = vcvtq_f32_u32(u32x4Src3);
                f32x4Nume0 = vmlaq_f32(f32x4Nume0, f32x4Src0, f32x4Coef);
                f32x4Nume1 = vmlaq_f32(f32x4Nume1, f32x4Src1, f32x4Coef);
                f32x4Nume2 = vmlaq_f32(f32x4Nume2, f32x4Src2, f32x4Coef);
                f32x4Nume3 = vmlaq_f32(f32x4Nume3, f32x4Src3, f32x4Coef);
            }

            // dst[dstX] = nume;
            float32x4_t f32x4Dst0 = f32x4Nume0;
            float32x4_t f32x4Dst1 = f32x4Nume1;
            float32x4_t f32x4Dst2 = f32x4Nume2;
            float32x4_t f32x4Dst3 = f32x4Nume3;
            vst1q_f32(&dst[dstX +  0], f32x4Dst0);
            vst1q_f32(&dst[dstX +  4], f32x4Dst1);
            vst1q_f32(&dst[dstX +  8], f32x4Dst2);
            vst1q_f32(&dst[dstX + 12], f32x4Dst3);
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

    void LinearResizerImpl<ArchNEON>::resizeX(const float * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            int32_t dstW = m_DstW;
            for ( int32_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = cvt_roundf32_u8(src[dstX]);
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
    void LinearResizerImpl<ArchNEON>::resizeXborder(
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
                dst[dstX] = cvt_roundf32_u8(src[0]);
                continue;
            }
            if ( mainEnd <= dstX ) {
                dst[dstX] = cvt_roundf32_u8(src[srcW - 1]);
                continue;
            }

            ptrdiff_t iCoef = getCoefXIndex(dstX);
            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                float   coef = coefs[iCoef];
                int32_t srcX = srcOX + i;
                sum   += src[srcX] * coef;
                iCoef += kVecStepX;
            }

            dst[dstX] = cvt_roundf32_u8(sum);
        }
    }

    //! resize horizontal (main loop)
    //!
    //! @param src    A row of source
    //! @param dst    A row of destination
    //! @param begin  Position of a first pixel
    //! @param end    Position of next of a last pixel
    void LinearResizerImpl<ArchNEON>::resizeXmain(
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
            //          nume        = 0;
            float32x4_t f32x4Nume0  = vdupq_n_f32(0);
            float32x4_t f32x4Nume1  = vdupq_n_f32(0);
            //          srcOX       = int32_t(floor((dstX+0.5) / scale - 0.5));
            int32x4_t   s32x4SrcOX0 = vld1q_s32(&indices[dstX + 0]);
            int32x4_t   s32x4SrcOX1 = vld1q_s32(&indices[dstX + 4]);

            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                //           srcX        = srcOX + i;
                int32x4_t    s32x4Offset = vdupq_n_s32(i);
                int32x4_t    s32x4SrcX0  = vaddq_s32(s32x4SrcOX0, s32x4Offset);
                int32x4_t    s32x4SrcX1  = vaddq_s32(s32x4SrcOX1, s32x4Offset);

                //           iNume      += src[srcX] * coefs[iCoef];
                float32x4_t  f32x4Src0   = gather(src, s32x4SrcX0);
                float32x4_t  f32x4Src1   = gather(src, s32x4SrcX1);
                float32x4_t  f32x4Coefs0 = vld1q_f32(&coefs[iCoef + 0]);
                float32x4_t  f32x4Coefs1 = vld1q_f32(&coefs[iCoef + 4]);

                // nume   += iNume;
                f32x4Nume0 = vmlaq_f32(f32x4Nume0, f32x4Src0, f32x4Coefs0);
                f32x4Nume1 = vmlaq_f32(f32x4Nume1, f32x4Src1, f32x4Coefs1);

                iCoef += kVecStepX;
            }

            // dst[dstX] = clamp<int>(0, 255, round(nume));
            float32x4_t  f32x4Dst0     = f32x4Nume0;
            float32x4_t  f32x4Dst1     = f32x4Nume1;
            uint8x8_t    u8x8Dst       = cvt_roundf32_u8(f32x4Dst0, f32x4Dst1);
            vst1_u8(&dst[dstX], u8x8Dst);

            // iCoef = dstX % tableSize;
            if ( iCoef == tableSize ) {
                iCoef = 0;
            }
        }
    }

    //! get index of m_TablesX from coordinate of destination
    ptrdiff_t LinearResizerImpl<ArchNEON>::getCoefXIndex(int32_t dstX)
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
    bool LinearResizerImpl_hasFeature<ArchNEON>()
    {
        return false;
    }

    template<>
    ILinearResizerImpl * LinearResizerImpl_new<ArchNEON>()
    {
        return NULL;
    }

}

#endif
