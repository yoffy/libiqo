#include <stdint.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <smmintrin.h>

#if defined(_OPENMP)
    #include <omp.h>
#endif

#include "libiqo/IQOLanczosResizer.hpp"


namespace {

    int getNumberOfProcs()
    {
#if defined(_OPENMP)
        return omp_get_num_procs();
#else
        return 1;
#endif
    }

    int getThreadNumber()
    {
#if defined(_OPENMP)
        return omp_get_thread_num();
#else
        return 0;
#endif
    }

    intptr_t alignFloor(intptr_t v, intptr_t alignment)
    {
        return v / alignment * alignment;
    }

    intptr_t alignCeil(intptr_t v, intptr_t alignment)
    {
        return (v + (alignment - 1)) / alignment * alignment;
    }

    template<typename T>
    T sinc(T x)
    {
        T fPi  = 3.14159265358979;
        T fPiX = fPi * x;
        return std::sin(fPiX) / fPiX;
    }

    template<typename T>
    T lanczos(int degree, T x)
    {
        T absX = std::fabs(x);
        if ( std::fmod(absX, T(1)) < T(1e-5) ) {
            return absX < T(1e-5) ? 1 : 0;
        }
        if ( degree <= absX ) {
            return 0;
        }
        return sinc(x) * sinc(x / degree);
    }

    //! @brief Set Lanczos table
    //! @param degree     Degree of Lanczos (ex. A=2 means Lanczos2)
    //! @param srcLen     Number of pixels of the source image
    //! @param dstLen     Number of pixels of the destination image
    //! @param dstOffset  The coordinate of the destination image
    //! @param pxScale  Scale of a pixel (ex. 2 when U plane of YUV420 image)
    //! @param n          Size of table
    //! @param fTable     The table
    //! @return Sum of the table
    //!
    //! Calculate Lanczos coefficients from `-degree` to `+degree`.
    //!
    //! n should be `2*degree` when up sampling.
    template<typename T>
    T setLanczosTable(
        int degree,
        intptr_t srcLen,
        intptr_t dstLen,
        intptr_t dstOffset,
        intptr_t pxScale,
        int n,
        T * fTable)
    {
        // down-sampling (ex. lanczos3)
        //
        // case decimal number of coefs:
        //        scale = 5:4
        // num of coefs = 3*5/4 = 3.75
        //
        //       start = -3
        //  num pixels =  4
        // -3           0
        //  o   o   o   o   o   o   o   o   o   o   o
        //  o    o    o    o    o    o    o    o    o
        //  +--------------+
        //
        //                       start = -3
        //                  num pixels =  4
        //                 -3           0
        //  o   o   o   o   o   o   o   o   o   o   o
        //  o    o    o    o    o    o    o    o    o
        //                 +--------------+
        //
        // case integer number of coefs:
        //        scale = 4:3
        // num of coefs = 3*4/3 = 4
        //
        //       start = -4 -> -3
        //  num pixels =  5 ->  4 (correct into num of coefs)
        // -4           0
        //  o  o  o  o  o  o  o  o  o
        //  o   o   o   o   o   o   o
        //  +-----------+
        //
        //                start = -3
        //           num pixels =  4
        //          -3        0
        //  o  o  o  o  o  o  o  o  o
        //  o   o   o   o   o   o   o
        //          +-----------+
        //
        // case integer number of coefs:
        //        scale = 8:3
        // num of coefs = 3*8/3 = 8
        //
        //       start = -8 -> -7
        //  num pixels =  9 ->  8 (correct into num of coefs)
        // -8             0
        //  o o o o o o o o o o o o o o o o
        //  o      o      o      o      o
        //  +-------------+

        //   o: center of a pixel (on a coordinate)
        //   |: boundary of pixels
        //
        // Theoretical space (aligned to center of a pixel):
        // start:     -degree + std::fmod((1 - srcOffset) * scale, 1.0)
        //             v
        //   src:    | o | o | o | o | o | o | o | o | o | o | o | o | o |
        //   dst:  |   o   |   o   |   o   |   o   |   o   |   o   |   o   |
        //             3       2       1       0       1       2       3
        //             ^
        //             degree
        //
        // Display space (aligned to boundary of pixels):
        // start:     (-degree - 0.5 + 0.5*scale) + std::fmod((1 - srcOffset) * scale, 1.0)
        //             v
        //   src:   |  o  |  o  |  o  |  o  |  o  |  o  |  o  |  o  |  o  |  o  |  o  |  o  |  o  |
        //   dst:   |  : o     |    o     |    o     |     o     |     o    |     o    |     o    |
        //         3.5 : 3          2          1           0           1          2          3
        //          ^  :
        // degree-0.5  0.5*scale

        // X is offset of Lanczos from center.
        // It will be source coordinate when up-sampling,
        // or, destination coordinate when down-sampling.
        double beginX = 0;
        if ( srcLen > dstLen ) {
            // down-sampling

            //----- easy solution -----
            //double scale = dstLen / double(srcLen);
            //double srcOffset = std::fmod(dstOffset / scale, 1.0);
            //double beginX = -degree + (-0.5 + 0.5*scale)*pxScale + std::fmod((1 - srcOffset) * scale * pxScale, 1.0);

            //----- more accurate -----
            // srcOffset = std::fmod(dstOffset / scale, 1.0)
            //           = (dstOffset * srcLen % dstLen) / dstLen;

            // -degree + (-0.5 + 0.5*scale)*pxScale
            // = -degree + (-0.5 + 0.5*dstLen/srcLen) * pxScale
            // = -degree - 0.5*pxScale + 0.5*dstLen*pxScale/srcLen

            // std::fmod((1 - srcOffset) * scale * pxScale, 1.0)
            // = std::fmod((1 - (dstOffset * srcLen % dstLen)/dstLen) * (dstLen/srcLen) * pxScale, 1.0)
            // = std::fmod((dstLen/srcLen - (dstOffset * srcLen % dstLen)/dstLen*(dstLen/srcLen)) * pxScale, 1.0)
            // = std::fmod((dstLen/srcLen - (dstOffset * srcLen % dstLen)/srcLen) * pxScale, 1.0)
            // = std::fmod(((dstLen - (dstOffset * srcLen % dstLen))/srcLen) * pxScale, 1.0)
            // = std::fmod( (dstLen - (dstOffset * srcLen % dstLen))         * pxScale, srcLen) / srcLen
            // = ((dstLen - dstOffset*srcLen%dstLen) * pxScale % srcLen) / srcLen
            int degFactor = std::max<int>(1, pxScale / degree);
            beginX =
                -degree*degFactor - 0.5*pxScale + 0.5*dstLen*pxScale/srcLen
                + ((dstLen - dstOffset * srcLen % dstLen) * pxScale % srcLen) / double(srcLen);
        } else {
            // up-sampling
            double srcOffset = std::fmod(dstOffset * srcLen / double(dstLen), 1.0);
            beginX = -degree + 1.0 - srcOffset;
            srcLen = dstLen; // scale = 1.0
            pxScale = 1;
        }

        T fSum = 0;

        for ( intptr_t i = 0; i < n; ++i ) {
            //     x = beginX + i * scale * pxScale
            double x = beginX + (i * dstLen * pxScale) / double(srcLen);
            T v = T(lanczos(degree, x));
            fTable[i] = v;
            fSum     += v;
        }

        return fSum;
    }

    template<typename T>
    T round(T x)
    {
        return std::floor(x + T(0.5));
    }

    template<typename T>
    T clamp(T lo, T hi, T v)
    {
        return std::max(lo, std::min(hi, v));
    }

    intptr_t gcd(intptr_t a, intptr_t b)
    {
        intptr_t r = a % b;

        while ( r ) {
            a = b;
            b = r;
            r = a % b;
        }

        return b;
    }

    intptr_t lcm(intptr_t a, intptr_t b)
    {
        return a * b / gcd(a, b);
    }

    __m128i cmpgt_epu8(__m128i a, __m128i b)
    {
        const __m128i k0x80 = _mm_set1_epi8(0x80);
        return _mm_cmpgt_epi8(_mm_xor_si128(a, k0x80), _mm_xor_si128(b, k0x80));
    }

    __m128i gather_epi16(const int16_t * s16src, __m128i u16x8Indices)
    {
        const __m128i k16     = _mm_set1_epi8(16);
        const __m128i k0x0100 = _mm_set1_epi16(0x0100); // 1, 0

        __m128i u8x16Result = _mm_setzero_si128();
        uint16_t iFirst = uint16_t(_mm_cvtsi128_si32(u16x8Indices));
        __m128i u16x8First0 = _mm_shufflelo_epi16(u16x8Indices, _MM_SHUFFLE(0, 0, 0, 0));
        __m128i u16x8First  = _mm_unpacklo_epi64(u16x8First0, u16x8First0);
        u16x8Indices = _mm_sub_epi16(u16x8Indices, u16x8First);
        // indices *= sizeof(uint16_t)
        __m128i u16x8Indices_0 = _mm_slli_epi16(u16x8Indices, 1);
        // u8x16Indices = { indices + 1, indices }
        __m128i u16x8Indices_1 = _mm_or_si128(u16x8Indices_0, _mm_slli_epi16(u16x8Indices_0, 8));
        __m128i u8x16Indices   = _mm_add_epi8(u16x8Indices_1, k0x0100);
        const uint8_t * src = reinterpret_cast<const uint8_t *>(s16src + iFirst);

        uint32_t b16IsDone = 0;
        __m128i  u8x16IsIn = cmpgt_epu8(k16, u8x16Indices);
        uint32_t b16IsIn   = _mm_movemask_epi8(u8x16IsIn);
        while ( b16IsIn & ~b16IsDone ) {
            __m128i u8x16Src  = _mm_loadu_si128((const __m128i*)src);
            __m128i u8x16Perm = _mm_shuffle_epi8(u8x16Src, u8x16Indices);
            u8x16Result  = _mm_blendv_epi8(u8x16Result, u8x16Perm, u8x16IsIn);
            u8x16Indices = _mm_sub_epi8(u8x16Indices, k16);
            src += 16;
            b16IsDone |= b16IsIn;
            u8x16IsIn  = cmpgt_epu8(k16, u8x16Indices);
            b16IsIn    = _mm_movemask_epi8(u8x16IsIn);
        }

        return u8x16Result;
    }

    __m128i mask_gather_epi16(const int16_t * s16src, __m128i s16x8Indices, __m128i u16x8Mask)
    {
        int16_t s16Indices[8];
        uint16_t s16Dst[8];
        uint32_t b16Mask = _mm_movemask_epi8(u16x8Mask);
        _mm_storeu_si128((__m128i*)s16Indices, s16x8Indices);
        for ( int i = 0; i < 8; ++i ) {
            s16Dst[i] = (b16Mask & 1) ? s16src[s16Indices[i]] : 0;
            b16Mask >>= 1;
        }
        return _mm_loadu_si128((const __m128i*)s16Dst);
    }


}

namespace iqo {

    class LanczosResizer::Impl
    {
    public:
        //! Constructor
        Impl(unsigned int degree, size_t srcW, size_t srcH, size_t dstW, size_t dstH, size_t pxScale);

        //! Run image resizing
        void resize(size_t srcSt, const uint8_t * src, size_t dstSt, uint8_t * __restrict dst);

    private:
        //! round(a / b)
        static int roundedDiv(int a, int b, int biasbit)
        {
            const int k0_5 = (1 << biasbit) / 2;
            return (a + k0_5) / b;
        }

        //! (int)round(a)
        static int cvtFixedToInt(int a, int biasbit)
        {
            const int k0_5 = (1 << biasbit) / 2;
            return (a + k0_5) >> biasbit;
        }

        //! (int16_t)round(a)
        static __m128i cvtFixedToInt(__m128i a)
        {
            // (a + kFixed0_5) / kBias
            __m128i s16x8tmp = _mm_add_epi16(a, _mm_set1_epi16(kFixed0_5));
            return _mm_srai_epi16(s16x8tmp, kBiasBit);
        }

        //! dst[i] = src[i] * kBias / srcSum (src will be broken)
        void adjustCoefs(
            float * srcBegin, const float * srcEnd,
            float srcSum,
            int bias,
            int16_t * dst);

        void resizeYborder(
            intptr_t srcSt, const uint8_t * src,
            intptr_t dstW, float * dst,
            intptr_t srcOY,
            const float * coefs);
        void resizeYmain(
            intptr_t srcSt, const uint8_t * src,
            intptr_t dstW, float * dst,
            intptr_t srcOY,
            const float * coefs);
        void resizeX(const int16_t * src, uint8_t * dst);
        void resizeXborder(
            const int16_t * src, uint8_t * dst,
            intptr_t begin, intptr_t end);
        void resizeXmain(
            const int16_t * src, uint8_t * dst,
            intptr_t begin, intptr_t end);

        enum {
            //! for SIMD
            kVecStepX  = 16, //!< int16x16
            kVecStepY  =  8, //!< float32x8

            //! for fixed point
            kBiasBit    = 6,
            kBias       = 1 << kBiasBit,
            kFixed1_0   = kBias,            //!< 1.0 in fixed point
            kFixed0_5   = kFixed1_0 / 2,    //!< 0.5 in fixed point

            kBias15Bit  = 15,
            kBias15     = 1 << kBias15Bit,  //!< for pmulhrsw
        };
        intptr_t m_SrcW;
        intptr_t m_SrcH;
        intptr_t m_DstW;
        intptr_t m_DstH;
        intptr_t m_NumCoefsX;
        intptr_t m_NumCoefsY;
        intptr_t m_NumCoordsX;
        intptr_t m_NumUnrolledCoordsX;
        intptr_t m_TablesXWidth;
        intptr_t m_NumCoordsY;
        std::vector<int16_t> m_TablesX_;    //!< m_TablesXWidth * m_NumCoordsX (unrolled)
        int16_t * m_TablesX;                //!< aligned
        std::vector<float> m_TablesY;     //!< Lanczos table * m_NumCoordsY
        std::vector<float> m_Work;
        std::vector<uint16_t> m_IndicesX;
    };

    LanczosResizer::LanczosResizer(
        unsigned int degree,
        size_t srcW,
        size_t srcH,
        size_t dstW,
        size_t dstH,
        size_t pxScale)
    {
        m_Impl = new Impl(degree, srcW, srcH, dstW, dstH, pxScale);
    }

    LanczosResizer::~LanczosResizer()
    {
        delete m_Impl;
    }

    void LanczosResizer::resize(size_t srcSt, const unsigned char * src, size_t dstSt, unsigned char * dst)
    {
        m_Impl->resize(srcSt, src, dstSt, dst);
    }


    //==================================================
    // LanczosResizer::Impl
    //==================================================

    // Constructor
    LanczosResizer::Impl::Impl(unsigned int degree, size_t srcW, size_t srcH, size_t dstW, size_t dstH, size_t pxScale)
    {
        m_SrcW = srcW;
        m_SrcH = srcH;
        m_DstW = dstW;
        m_DstH = dstH;

        // setup coefficients
        if ( m_SrcW <= m_DstW ) {
            // horizontal: up-sampling
            m_NumCoefsX = 2 * degree;
        } else {
            // vertical: down-sampling
            // n = 2*degree / scale
            size_t degree2 = std::max<size_t>(1, degree / pxScale);
            m_NumCoefsX = 2 * intptr_t(std::ceil((degree2 * m_SrcW) / double(m_DstW)));
        }
        if ( m_SrcH <= m_DstH ) {
            // vertical: up-sampling
            m_NumCoefsY = 2 * degree;
        } else {
            // vertical: down-sampling
            // n = 2*degree / scale
            size_t degree2 = std::max<size_t>(1, degree / pxScale);
            m_NumCoefsY = 2 * intptr_t(std::ceil((degree2 * m_SrcH) / double(m_DstH)));
        }
        m_NumCoordsX = m_DstW / gcd(m_SrcW, m_DstW);
        m_NumUnrolledCoordsX = std::min(alignCeil(m_DstW, kVecStepX), lcm(m_NumCoordsX, kVecStepX));
        m_TablesXWidth = kVecStepX * m_NumCoefsX;
        m_NumCoordsY = m_DstH / gcd(m_SrcH, m_DstH);
        m_TablesX_.reserve(m_TablesXWidth * m_NumCoordsX + kVecStepX);
        m_TablesX_.resize(m_TablesXWidth * m_NumCoordsX + kVecStepX);
        m_TablesX = (int16_t *)alignCeil(intptr_t(&m_TablesX_[0]), sizeof(*m_TablesX) * kVecStepX);
        m_TablesY.reserve(m_NumCoefsY * m_NumCoordsY);
        m_TablesY.resize(m_NumCoefsY * m_NumCoordsY);

        // X coefs
        std::vector<int16_t> tablesX(m_NumCoefsX * m_NumCoordsX);
        std::vector<float> tableX(m_NumCoefsX);
        for ( intptr_t dstX = 0; dstX < m_NumCoordsX; ++dstX ) {
            int16_t * table = &tablesX[dstX * m_NumCoefsX];
            double sumCoefs = setLanczosTable(degree, m_SrcW, m_DstW, dstX, pxScale, m_NumCoefsX, &tableX[0]);
            adjustCoefs(&tableX[0], &tableX[m_NumCoefsX], sumCoefs, kBias15, &table[0]);
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
        intptr_t tblW = m_TablesXWidth;
        intptr_t nCoords = m_NumCoordsX;
        for ( intptr_t row = 0; row < nCoords; ++row ) {
            for ( intptr_t col = 0; col < tblW; ++col ) {
                intptr_t iCoef = col/kVecStepX;
                intptr_t srcX = (col%kVecStepX + row) % nCoords;
                intptr_t i = iCoef + srcX*m_NumCoefsX;
                m_TablesX[col + m_TablesXWidth*row] = tablesX[i];
            }
        }

        // Y coefs
        std::vector<float> tablesY(m_NumCoefsY);
        for ( intptr_t dstY = 0; dstY < m_NumCoordsY; ++dstY ) {
            float * table = &m_TablesY[dstY * m_NumCoefsY];
            double sumCoefs = setLanczosTable(degree, m_SrcH, m_DstH, dstY, pxScale, m_NumCoefsY, table);
            for ( intptr_t i = 0; i < m_NumCoefsY; ++i ) {
                table[i] /= sumCoefs;
            }
        }

        // allocate workspace
        m_Work.reserve(m_SrcW * getNumberOfProcs());
        m_Work.resize(m_SrcW * getNumberOfProcs());

        // calc indices
        intptr_t alignedDstW = alignCeil(m_DstW, kVecStepX);
        m_IndicesX.reserve(alignedDstW);
        m_IndicesX.resize(alignedDstW);
        for ( intptr_t dstX = 0; dstX < alignedDstW; ++dstX ) {
            int32_t srcOX = dstX * m_SrcW / m_DstW + 1;
            m_IndicesX[dstX] = srcOX;
        }
    }

    //! dst[i] = src[i] * kBias / srcSum (src will be broken)
    void LanczosResizer::Impl::adjustCoefs(
        float * __restrict srcBegin, const float * srcEnd,
        float srcSum,
        int bias,
        int16_t * __restrict dst)
    {
        const int k1_0 = bias;
        size_t numCoefs = srcEnd - srcBegin;
        int dstSum = 0;

        for ( size_t i = 0; i < numCoefs; ++i ) {
            dst[i] = round(srcBegin[i] * bias / srcSum);
            dstSum += dst[i];
        }
        while ( dstSum < k1_0 ) {
            size_t i = std::distance(&srcBegin[0], std::max_element(&srcBegin[0], &srcBegin[m_NumCoefsX]));
            dst[i]++;
            srcBegin[i] = 0;
            dstSum++;
        }
        while ( dstSum > k1_0 ) {
            size_t i = std::distance(&srcBegin[0], std::max_element(&srcBegin[0], &srcBegin[m_NumCoefsX]));
            dst[i]--;
            srcBegin[i] = 0;
            dstSum--;
        }
    }

    void LanczosResizer::Impl::resize(size_t srcSt, const uint8_t * src, size_t dstSt, uint8_t * __restrict dst)
    {
        // resize

        if ( m_SrcH == m_DstH ) {
#pragma omp parallel for
            for ( intptr_t y = 0; y < m_SrcH; ++y ) {
                float * work = &m_Work[getThreadNumber() * m_SrcW];
                for ( intptr_t x = 0; x < m_SrcW; ++x ) {
                    m_Work[m_SrcW * y + x] = src[srcSt * y + x] * kBias;
                }
                resizeX(work, &dst[dstSt * y]);
            }
        } else {
            // vertical
            intptr_t numCoefsOn2 = m_NumCoefsY / 2;
            // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstH / double(m_SrcH))
            intptr_t mainBegin = ((numCoefsOn2 - 1) * m_DstH + m_SrcH-1) / m_SrcH;
            intptr_t mainEnd = std::max<intptr_t>(0, (m_SrcH - numCoefsOn2) * m_DstH / m_SrcH);
            const float * tablesY = &m_TablesY[0];

#pragma omp parallel for
            for ( intptr_t dstY = 0; dstY < mainBegin; ++dstY ) {
                float * work = &m_Work[getThreadNumber() * m_SrcW];
                intptr_t srcOY = dstY * m_SrcH / m_DstH + 1;
                const float * coefs = &tablesY[dstY % m_NumCoordsY * m_NumCoefsY];
                resizeYborder(
                    srcSt, &src[0],
                    m_SrcW, work,
                    srcOY,
                    coefs);
                resizeX(work, &dst[dstSt * dstY]);
            }
#pragma omp parallel for
            for ( intptr_t dstY = mainBegin; dstY < mainEnd; ++dstY ) {
                float * work = &m_Work[getThreadNumber() * m_SrcW];
                intptr_t srcOY = dstY * m_SrcH / m_DstH + 1;
                const float * coefs = &tablesY[dstY % m_NumCoordsY * m_NumCoefsY];
                resizeYmain(
                    srcSt, &src[0],
                    m_SrcW, work,
                    srcOY,
                    coefs);
                resizeX(work, &dst[dstSt * dstY]);
            }
#pragma omp parallel for
            for ( intptr_t dstY = mainEnd; dstY < m_DstH; ++dstY ) {
                float * work = &m_Work[getThreadNumber() * m_SrcW];
                intptr_t srcOY = dstY * m_SrcH / m_DstH + 1;
                const float * coefs = &tablesY[dstY % m_NumCoordsY * m_NumCoefsY];
                resizeYborder(
                    srcSt, &src[0],
                    m_SrcW, work,
                    srcOY,
                    coefs);
                resizeX(work, &dst[dstSt * dstY]);
            }
        }
    }

    //! resize vertical (border loop)
    //!
    //! @param srcSt  Stride in src (in byte)
    //! @param src    A row of source
    //! @param dst    A row of destination (multiplied by kBias)
    //! @param srcOY  The origin of current line
    //! @param coefs  The coefficients (multiplied by kBias)
    void LanczosResizer::Impl::resizeYborder(
        intptr_t srcSt, const uint8_t * src,
        intptr_t dstW, float * __restrict dst,
        intptr_t srcOY,
        const float * coefs)
    {
        intptr_t numCoefsOn2 = m_NumCoefsY / 2;
        intptr_t vecLen = alignFloor(dstW, kVecStepY);
        intptr_t numCoefsY = m_NumCoefsY;
        intptr_t srcH = m_SrcH;

        for ( intptr_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            // nume = 0;
            __m128 f32x4Nume0 = _mm_setzero_ps();
            __m128 f32x4Nume1 = _mm_setzero_ps();
            float deno = 0;

            for ( intptr_t i = 0; i < numCoefsY; ++i ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                if ( 0 <= srcY && srcY < srcH ) {
                    // coef = coefs[i];
                    __m128  f32x4Coef = _mm_set1_ps(coefs[i]);
                    // nume += src[dstX + srcSt*srcY] * coef;
                    __m128i u8x8Src     = _mm_loadl_pi(_mm_setzero_ps(), (const __m64 *)&src[dstX + srcSt*srcY]);
                    __m128i u8x4Src0    = u8x8Src;
                    __m128i u8x4Src1    = _mm_shuffle_epi32(u8x8Src, _MM_SHUFFLE(3, 3, 3, 1));
                    __m128  f32x4Src0   = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(u8x4Src0));
                    __m128  f32x4Src1   = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(u8x4Src1));
                    f32x4Nume0 = _mm_add_ps(_mm_mul_ps(f32x4Src0, f32x4Coef), f32x4Nume0);
                    f32x4Nume1 = _mm_add_ps(_mm_mul_ps(f32x4Src1, f32x4Coef), f32x4Nume1);
                    deno += coefs[i];
                }
            }

            // dst[dstX] = nume / deno;
            __m128 f32x4Deno = _mm_set1_ps(deno);
            __m128 f32x4Dst0 = _mm_mul_ps(f32x4Nume0, _mm_rcp_ps(f32x4Deno));
            __m128 f32x4Dst1 = _mm_mul_ps(f32x4Nume1, _mm_rcp_ps(f32x4Deno));
            _mm_storeu_ps(&dst[dstX + 0], f32x4Dst0);
            _mm_storeu_ps(&dst[dstX + 8], f32x4Dst1);
        }

        for ( intptr_t dstX = vecLen; dstX < dstW; dstX++ ) {
            float nume = 0;
            float deno = 0;

            for ( intptr_t i = 0; i < numCoefsY; ++i ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                float    coef = coefs[i];
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
    //! @param dst    A row of destination (multiplied by kBias)
    //! @param srcOY  The origin of current line
    //! @param coefs  The coefficients (multiplied by kBias)
    void LanczosResizer::Impl::resizeYmain(
        intptr_t srcSt, const uint8_t * src,
        intptr_t dstW, float * __restrict dst,
        intptr_t srcOY,
        const float * coefs)
    {
        intptr_t numCoefsOn2 = m_NumCoefsY / 2;
        intptr_t vecLen = alignFloor(dstW, kVecStepY);
        intptr_t numCoefsY = m_NumCoefsY;

        for ( intptr_t dstX = 0; dstX < vecLen; dstX += kVecStepY ) {
            // nume = 0;
            __m128 f32x4Nume0 = _mm_setzero_ps();
            __m128 f32x4Nume1 = _mm_setzero_ps();

            for ( intptr_t i = 0; i < numCoefsY; ++i ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                // coef = coefs[i];
                __m128  f32x4Coef = _mm_set1_ps(coefs[i]);
                // nume += src[dstX + srcSt*srcY] * coef;
                __m128i u8x8Src     = _mm_loadl_pi(_mm_setzero_ps(), (const __m64 *)&src[dstX + srcSt*srcY]);
                __m128i u8x4Src0    = u8x8Src;
                __m128i u8x4Src1    = _mm_shuffle_epi32(u8x8Src, _MM_SHUFFLE(3, 3, 3, 1));
                __m128  f32x4Src0   = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(u8x4Src0));
                __m128  f32x4Src1   = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(u8x4Src1));
                f32x4Nume0 = _mm_add_ps(_mm_mul_ps(f32x4Src0, f32x4Coef), f32x4Nume0);
                f32x4Nume1 = _mm_add_ps(_mm_mul_ps(f32x4Src1, f32x4Coef), f32x4Nume1);
            }

            // dst[dstX] = nume;
            __m128 f32x4Dst0 = f32x4Nume0;
            __m128 f32x4Dst1 = f32x4Nume1;
            _mm_storeu_ps(&dst[dstX + 0], f32x4Dst0);
            _mm_storeu_ps(&dst[dstX + 8], f32x4Dst1);
        }

        for ( intptr_t dstX = vecLen; dstX < dstW; dstX++ ) {
            float nume = 0;

            for ( intptr_t i = 0; i < numCoefsY; ++i ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                float    coef = coefs[i];
                nume += src[dstX + srcSt * srcY] * coef;
            }

            dst[dstX] = nume;
        }
    }

    void LanczosResizer::Impl::resizeX(const float * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            intptr_t dstW = m_DstW;
            for ( intptr_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = clamp<int>(0, 255, round(src[dstX]);
            }
            return;
        }

        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstW / double(m_SrcW))
        intptr_t mainBegin = alignCeil(((numCoefsOn2 - 1) * m_DstW + m_SrcW-1) / m_SrcW, kVecStepX);
        intptr_t mainEnd = std::max<intptr_t>(0, (m_SrcW - numCoefsOn2) * m_DstW / m_SrcW);
        intptr_t mainLen = alignFloor(mainEnd - mainBegin, kVecStepX);
        mainEnd = mainBegin + mainLen;

        resizeXborder(src, dst, 0,         mainBegin);
        resizeXmain  (src, dst, mainBegin, mainEnd);
        resizeXborder(src, dst, mainEnd,   m_DstW);
    }

    void LanczosResizer::Impl::resizeXborder(
        const * src, uint8_t * __restrict dst,
        intptr_t begin, intptr_t end)
    {
        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        const int16_t * coefs = &m_TablesX[0];
        const uint16_t * indices = &m_IndicesX[0];
        intptr_t tableSize = m_TablesXWidth * m_NumCoordsX;
        intptr_t numCoefsX = m_NumCoefsX;
        intptr_t dstW = m_DstW;

        __m128i s16x8k_1  = _mm_set1_epi16(-1);
        __m128i s16x8SrcW = _mm_set1_epi16(m_SrcW);
        intptr_t iCoef = begin / kVecStepX % m_NumCoordsX * m_TablesXWidth;
        for ( intptr_t dstX = begin; dstX < end; dstX += kVecStepX ) {
            // nume             = 0;
            __m128i s16x8Nume0  = _mm_setzero_si128();
            __m128i s16x8Nume1  = _mm_setzero_si128();
            // deno             = 0;
            __m128i s16x8Deno0  = _mm_setzero_si128();
            __m128i s16x8Deno1  = _mm_setzero_si128();
            // srcOX            = floor(dstX / scale) + 1;
            __m128i u16x8SrcOX0 = _mm_loadu_si128((const __m128i*)&indices[dstX + 0]);
            __m128i u16x8SrcOX1 = _mm_loadu_si128((const __m128i*)&indices[dstX + 8]);

            for ( intptr_t i = 0; i < numCoefsX; ++i ) {
                // srcX             = srcOX - numCoefsOn2 + i;
                __m128i s16x8Offset = _mm_set1_epi16(i - numCoefsOn2);
                __m128i s16x8SrcX0  = _mm_add_epi16(u16x8SrcOX0, s16x8Offset);
                __m128i s16x8SrcX1  = _mm_add_epi16(u16x8SrcOX1, s16x8Offset);
                // if ( 0 <= srcX && srcX < m_SrcW )
                __m128i u16x8Mask0  = _mm_and_si128(_mm_cmpgt_epi16(s16x8SrcX0, s16x8k_1), _mm_cmpgt_epi16(s16x8SrcW, s16x8SrcX0));
                __m128i u16x8Mask1  = _mm_and_si128(_mm_cmpgt_epi16(s16x8SrcX1, s16x8k_1), _mm_cmpgt_epi16(s16x8SrcW, s16x8SrcX1));
                // iNume            += src[srcX] * coefs[iCoef];
                __m128i s16x8Src0   = mask_gather_epi16((const int16_t *)src, s16x8SrcX0, u16x8Mask0);
                __m128i s16x8Src1   = mask_gather_epi16((const int16_t *)src, s16x8SrcX1, u16x8Mask1);
                __m128i s16x8Coefs0 = _mm_load_si128((const __m128i *)&coefs[iCoef + 0]);
                __m128i s16x8Coefs1 = _mm_load_si128((const __m128i *)&coefs[iCoef + 8]);
                __m128i s16x8iNume0 = _mm_mulhrs_epi16(s16x8Src0, s16x8Coefs0);
                __m128i s16x8iNume1 = _mm_mulhrs_epi16(s16x8Src1, s16x8Coefs1);
                // nume   += iNume;
                s16x8Nume0 = _mm_add_epi16(s16x8Nume0, s16x8iNume0);
                s16x8Nume1 = _mm_add_epi16(s16x8Nume1, s16x8iNume1);
                // deno   += coefs[iCoef] >> (kBias15Bit - kBiasBit);
                s16x8Deno0 = _mm_add_epi16(s16x8Deno0, _mm_srli_epi16(s16x8Coefs0, kBias15Bit - kBiasBit));
                s16x8Deno1 = _mm_add_epi16(s16x8Deno1, _mm_srli_epi16(s16x8Coefs1, kBias15Bit - kBiasBit));

                iCoef += kVecStepX;
            }

            // dst[dstX] = round(nume / (deno*kBias))
            int16_t s16Numes[kVecStepX];
            int16_t s16Denos[kVecStepX];
            _mm_storeu_si128((__m128i*)&s16Numes[0], s16x8Nume0);
            _mm_storeu_si128((__m128i*)&s16Numes[8], s16x8Nume1);
            _mm_storeu_si128((__m128i*)&s16Denos[0], s16x8Deno0);
            _mm_storeu_si128((__m128i*)&s16Denos[8], s16x8Deno0);
            for ( intptr_t i = 0; i < kVecStepX && dstX+i < dstW; ++i ) {
                dst[dstX + i] = clamp<int16_t>(0, 255, roundedDiv(s16Numes[i], s16Denos[i], kBiasBit));
            }

            // iCoef = dstX % tableSize;
            if ( iCoef == tableSize ) {
                iCoef = 0;
            }
        }
    }

    //! resize horizontal (main loop)
    //!
    //! @param src    A row of source (multiplied by kBias)
    //! @param dst    A row of destination
    //! @param begin  Position of a first pixel
    //! @param end    Position of next of a last pixel
    void LanczosResizer::Impl::resizeXmain(
        const int16_t * src, uint8_t * __restrict dst,
        intptr_t begin, intptr_t end)
    {
        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        const int16_t * coefs = &m_TablesX[0];
        const uint16_t * indices = &m_IndicesX[0];
        intptr_t tableSize = m_TablesXWidth * m_NumCoordsX;
        intptr_t numCoefsX = m_NumCoefsX;

        intptr_t iCoef = begin / kVecStepX % m_NumCoordsX * m_TablesXWidth;
        for ( intptr_t dstX = begin; dstX < end; dstX += kVecStepX ) {
            // nume             = 0;
            __m128i s16x8Nume0  = _mm_setzero_si128();
            __m128i s16x8Nume1  = _mm_setzero_si128();
            // srcOX            = floor(dstX / scale) + 1;
            __m128i u16x8SrcOX0 = _mm_loadu_si128((const __m128i*)&indices[dstX + 0]);
            __m128i u16x8SrcOX1 = _mm_loadu_si128((const __m128i*)&indices[dstX + 8]);

            for ( intptr_t i = 0; i < numCoefsX; ++i ) {
                // srcX             = srcOX - numCoefsOn2 + i;
                __m128i s16x8Offset = _mm_set1_epi16(i - numCoefsOn2);
                __m128i u16x8SrcX0  = _mm_add_epi16(u16x8SrcOX0, s16x8Offset);
                __m128i u16x8SrcX1  = _mm_add_epi16(u16x8SrcOX1, s16x8Offset);
                // iNume            += src[srcX] * coefs[iCoef];
                __m128i s16x8Src0   = gather_epi16((const int16_t *)src, u16x8SrcX0);
                __m128i s16x8Src1   = gather_epi16((const int16_t *)src, u16x8SrcX1);
                __m128i s16x8Coefs0 = _mm_load_si128((const __m128i *)&coefs[iCoef + 0]);
                __m128i s16x8Coefs1 = _mm_load_si128((const __m128i *)&coefs[iCoef + 8]);
                __m128i s16x8iNume0 = _mm_mulhrs_epi16(s16x8Src0, s16x8Coefs0);
                __m128i s16x8iNume1 = _mm_mulhrs_epi16(s16x8Src1, s16x8Coefs1);
                // nume   += iNume
                s16x8Nume0 = _mm_add_epi16(s16x8Nume0, s16x8iNume0);
                s16x8Nume1 = _mm_add_epi16(s16x8Nume1, s16x8iNume1);

                iCoef += kVecStepX;
            }

            // dst[dstX] = clamp<int16_t>(0, 255, cvtFixedToInt(nume, kBiasBit));
            __m128i s16x8Dst0 = cvtFixedToInt(s16x8Nume0);
            __m128i s16x8Dst1 = cvtFixedToInt(s16x8Nume1);
            __m128i u8x16Dst  = _mm_packus_epi16(s16x8Dst0, s16x8Dst1);
            _mm_storeu_si128((__m128i*)&dst[dstX], u8x16Dst);

            // iCoef = dstX % tableSize;
            if ( iCoef == tableSize ) {
                iCoef = 0;
            }
        }
    }

}
