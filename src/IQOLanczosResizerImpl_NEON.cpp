#include "IQOLanczosResizerImpl.hpp"


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

    float32x4_t gather(const float * src, int32x4_t indices)
    {
        float32x4_t v = float32x4_t();
        v = vld1q_lane_f32(&src[vgetq_lane_s32(indices, 0)], v, 0);
        v = vld1q_lane_f32(&src[vgetq_lane_s32(indices, 1)], v, 1);
        v = vld1q_lane_f32(&src[vgetq_lane_s32(indices, 2)], v, 2);
        v = vld1q_lane_f32(&src[vgetq_lane_s32(indices, 3)], v, 3);
        return v;
    }

    float32x4_t mask_gather(const float * src, int32x4_t indices, uint32x4_t mask)
    {
        float32x4_t v = vdupq_n_f32(0);
        if ( vgetq_lane_u32(mask, 0) ) v = vld1q_lane_f32(&src[vgetq_lane_s32(indices, 0)], v, 0);
        if ( vgetq_lane_u32(mask, 1) ) v = vld1q_lane_f32(&src[vgetq_lane_s32(indices, 1)], v, 1);
        if ( vgetq_lane_u32(mask, 2) ) v = vld1q_lane_f32(&src[vgetq_lane_s32(indices, 2)], v, 2);
        if ( vgetq_lane_u32(mask, 3) ) v = vld1q_lane_f32(&src[vgetq_lane_s32(indices, 3)], v, 3);
        return v;
    }

    //! reciprocal (16-bit precision)
    float32x4_t rcp16(float32x4_t v)
    {
        // precision of VRECPE is only 8-bit
        float32x4_t f32x4Rcp = vrecpeq_f32(v);
        // 16-bit
        f32x4Rcp = vmulq_f32(vrecpsq_f32(v, f32x4Rcp), f32x4Rcp);
        return f32x4Rcp;
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
    class LanczosResizerImpl<ArchNEON> : public ILanczosResizerImpl
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
            kVecStepX  =  8, //!< float32x4_t x 2
            kVecStepY  = 16, //!< float32x4_t x 4
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
    ILanczosResizerImpl * LanczosResizerImpl_new<ArchNEON>()
    {
        return new LanczosResizerImpl<ArchNEON>();
    }


    // Constructor
    void LanczosResizerImpl<ArchNEON>::init(
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

    void LanczosResizerImpl<ArchNEON>::resize(
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
    void LanczosResizerImpl<ArchNEON>::resizeYborder(
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
            //          nume       = 0;
            float32x4_t f32x4Nume0 = vdupq_n_f32(0);
            float32x4_t f32x4Nume1 = vdupq_n_f32(0);
            float32x4_t f32x4Nume2 = vdupq_n_f32(0);
            float32x4_t f32x4Nume3 = vdupq_n_f32(0);
            float deno = 0;

            for ( int32_t i = 0; i < numCoefsY; ++i ) {
                int32_t srcY = srcOY - numCoefsOn2 + i;

                if ( 0 <= srcY && srcY < srcH ) {
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

                    deno += coefs[i];
                }
            }

            // dst[dstX] = nume / deno;
            float32x4_t f32x4RcpDeno    = rcp16(vdupq_n_f32(deno));
            float32x4_t f32x4Dst0       = vmulq_f32(f32x4Nume0, f32x4RcpDeno);
            float32x4_t f32x4Dst1       = vmulq_f32(f32x4Nume1, f32x4RcpDeno);
            float32x4_t f32x4Dst2       = vmulq_f32(f32x4Nume2, f32x4RcpDeno);
            float32x4_t f32x4Dst3       = vmulq_f32(f32x4Nume3, f32x4RcpDeno);
            vst1q_f32(&dst[dstX +  0], f32x4Dst0);
            vst1q_f32(&dst[dstX +  4], f32x4Dst1);
            vst1q_f32(&dst[dstX +  8], f32x4Dst2);
            vst1q_f32(&dst[dstX + 12], f32x4Dst3);
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
    void LanczosResizerImpl<ArchNEON>::resizeYmain(
        ptrdiff_t srcSt, const uint8_t * src,
        int32_t dstW, float * __restrict dst,
        int32_t srcOY,
        const float * coefs
    ) {
        int32_t numCoefsOn2 = m_NumCoefsY / 2;
        int32_t vecLen = alignFloor<int32_t>(dstW, kVecStepY);
        int32_t numCoefsY = m_NumCoefsY;

        for ( int32_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            //          nume       = 0;
            float32x4_t f32x4Nume0 = vdupq_n_f32(0);
            float32x4_t f32x4Nume1 = vdupq_n_f32(0);
            float32x4_t f32x4Nume2 = vdupq_n_f32(0);
            float32x4_t f32x4Nume3 = vdupq_n_f32(0);

            for ( int32_t i = 0; i < numCoefsY; ++i ) {
                int32_t     srcY        = srcOY - numCoefsOn2 + i;

                //          coef        = coefs[i];
                float32x4_t f32x4Coef   = vdupq_n_f32(coefs[i]);

                //          nume       += src[dstX + srcSt*srcY] * coef;
                uint8x16_t  u8x16Src    = vld1q_u8(&src[dstX + srcSt*srcY]);
#if defined(__GNUC__)
                __builtin_prefetch(&src[dstX + srcSt*(srcY + numCoefsY)], 0, 2); // read, L2
#endif
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
                int32_t srcY = srcOY - numCoefsOn2 + i;
                float   coef = coefs[i];
                nume += src[dstX + srcSt * srcY] * coef;
            }

            dst[dstX] = nume;
        }
    }

    void LanczosResizerImpl<ArchNEON>::resizeX(const float * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            int32_t dstW = m_DstW;
            for ( int32_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = cvt_roundf32_u8(src[dstX]);
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
    void LanczosResizerImpl<ArchNEON>::resizeXborder(
        const float * src, uint8_t * __restrict dst,
        int32_t begin, int32_t end
    ) {
        int32_t numCoefsOn2 = m_NumCoefsX / 2;
        const float * coefs = &m_TablesX[0];
        const int32_t * indices = &m_IndicesX[0];
        ptrdiff_t tableWidth = ptrdiff_t(m_TablesXWidth);

        for ( ptrdiff_t dstX = begin; dstX < end; ++dstX ) {
            ptrdiff_t iCoef = (dstX % kVecStepX) + (dstX / kVecStepX % m_NumUnrolledCoordsX * tableWidth);
            //      srcOX = floor(dstX / scale) + 1;
            int32_t srcOX = indices[dstX];
            float nume = 0;
            float deno = 0;

            for ( int32_t i = 0; i < m_NumCoefsX; ++i ) {
                int32_t srcX = srcOX - numCoefsOn2 + i;
                if ( 0 <= srcX && srcX < m_SrcW ) {
                    float coef = coefs[i];
                    nume += src[srcX] * coef;
                    deno += coef;
                }
                iCoef += kVecStepX;
            }

            dst[dstX] = cvt_roundf32_u8(nume/deno);
        }
    }

    //! resize horizontal (main loop)
    //!
    //! @param src    A row of source
    //! @param dst    A row of destination
    //! @param begin  Position of a first pixel
    //! @param end    Position of next of a last pixel
    void LanczosResizerImpl<ArchNEON>::resizeXmain(
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
            //          nume        = 0;
            float32x4_t f32x4Nume0  = vdupq_n_f32(0);
            float32x4_t f32x4Nume1  = vdupq_n_f32(0);
            //          srcOX       = floor(dstX / scale) + 1;
            int32x4_t   s32x4SrcOX0 = vld1q_s32(&indices[dstX + 0]);
            int32x4_t   s32x4SrcOX1 = vld1q_s32(&indices[dstX + 4]);

            for ( int32_t i = 0; i < numCoefsX; ++i ) {
                //           srcX        = srcOX - numCoefsOn2 + i;
                int32x4_t    s32x4Offset = vdupq_n_s32(i - numCoefsOn2);
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

}

#else

namespace iqo {

    template<>
    ILanczosResizerImpl * LanczosResizerImpl_new<ArchNEON>()
    {
        return NULL;
    }

}

#endif
