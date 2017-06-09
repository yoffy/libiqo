#include <stdint.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <immintrin.h>

#include "libiqo/IQOLanczosResizer.hpp"


namespace {

    inline intptr_t alignFloor(intptr_t v, intptr_t alignment)
    {
        return v / alignment * alignment;
    }

    inline intptr_t alignCeil(intptr_t v, intptr_t alignment)
    {
        return (v + (alignment - 1)) / alignment * alignment;
    }

    template<typename T>
    inline T sinc(T x)
    {
        T fPi  = 3.14159265358979;
        T fPiX = fPi * x;
        return std::sin(fPiX) / fPiX;
    }

    template<typename T>
    inline T lanczos(int degree, T x)
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

    //! linear integer interpolation
    class LinearIterator
    {
    public:
        LinearIterator(intptr_t dx, intptr_t dy)
        {
            m_DX = dx;
            m_DY = dy;
            m_X = 0;
            m_Y = 0;
        }

        //! get y
        intptr_t operator*() const
        {
            return m_Y;
        }

        //! ++x
        LinearIterator & operator++()
        {
            advance();
            return *this;
        }

        //! x++
        LinearIterator operator++(int)
        {
            LinearIterator tmp(*this);
            advance();
            return tmp;
        }

    private:
        void advance()
        {
            m_X += m_DY;
            while ( m_X >= m_DX ) {
                ++m_Y;
                m_X -= m_DX;
            }
        }

        intptr_t m_DX;
        intptr_t m_DY;
        intptr_t m_X;
        intptr_t m_Y;
    };

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
        static int16_t roundedDiv(int16_t a, int16_t b)
        {
            return (a + kFixed0_5) / b;
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
            float * __restrict srcBegin, const float * srcEnd,
            float srcSum,
            int16_t * __restrict dst);

        void resizeYborder(
            intptr_t srcSt, const uint8_t * src,
            intptr_t dstW, int16_t * __restrict dst,
            intptr_t srcOY,
            const int16_t * coefs,
            int16_t * __restrict nume, int16_t * __restrict deno);
        void resizeYmain(
            intptr_t srcSt, const uint8_t * src,
            intptr_t dstW, int16_t * __restrict dst,
            intptr_t srcOY,
            const int16_t * coefs);
        void resizeX(const int16_t * src, uint8_t * __restrict dst);
        void resizeXborder(
            const int16_t * src, uint8_t * __restrict dst,
            intptr_t begin, intptr_t end);
        void resizeXmain(
            const int16_t * src, uint8_t * dst,
            intptr_t begin, intptr_t end);

        enum {
            //! for SIMD
            kVecStep  = 16, //!< int16x16
            //! for fixed point
            kBiasBit  = 6,
            kBias     = 1 << kBiasBit,
            kFixed1_0 = kBias,          //!< 1.0 in fixed point
            kFixed0_5 = kFixed1_0 / 2,  //!< 0.5 in fixed point
        };
        intptr_t m_SrcW;
        intptr_t m_SrcH;
        intptr_t m_DstW;
        intptr_t m_DstH;
        intptr_t m_NumCoefsX;
        intptr_t m_NumCoefsY;
        intptr_t m_NumTablesX;
        intptr_t m_NumTablesX2;
        intptr_t m_NumTablesY;
        std::vector<int16_t> m_TablesX;   //!< Lanczos table * m_NumTablesX
        std::vector<int16_t> m_TablesX2;  //!< Transposed
        std::vector<int16_t> m_TablesY;   //!< Lanczos table * m_NumTablesY
        std::vector<int16_t> m_Work;
        std::vector<int16_t> m_Nume;
        std::vector<int16_t> m_Deno;
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
        m_NumTablesX = m_DstW / gcd(m_SrcW, m_DstW);
        m_NumTablesX2 = lcm(m_NumTablesX, kVecStep);
        m_NumTablesY = m_DstH / gcd(m_SrcH, m_DstH);
        m_TablesX.reserve(m_NumCoefsX * m_NumTablesX);
        m_TablesX.resize(m_NumCoefsX * m_NumTablesX);
        m_TablesX2.reserve(m_NumTablesX2 * m_NumCoefsX);
        m_TablesX2.resize(m_NumTablesX2 * m_NumCoefsX);
        m_TablesY.reserve(m_NumCoefsY * m_NumTablesY);
        m_TablesY.resize(m_NumCoefsY * m_NumTablesY);

        // X coefs
        std::vector<float> tablesX(m_NumCoefsX);
        for ( intptr_t dstX = 0; dstX < m_NumTablesX; ++dstX ) {
            int16_t * table = &m_TablesX[dstX * m_NumCoefsX];
            double sumCoefs = setLanczosTable(degree, m_SrcW, m_DstW, dstX, pxScale, m_NumCoefsX, &tablesX[0]);
            adjustCoefs(&tablesX[0], &tablesX[m_NumCoefsX], sumCoefs, &table[0]);
        }
        // transpose and unroll X coefs
        //
        //  tables: A0A1A2A3
        //          B0B1B2B3
        //          C0C1C2C3
        //
        // tables2: A0B0C0A0 B0C0A0B0 C0A0B0C0 (align to SIMD unit)
        //                      :
        //          A3B3C3A3 B3C3A3B3 C3A3B3C3
        for ( intptr_t iCoef = 0; iCoef < m_NumCoefsX; ++iCoef ) {
            for ( intptr_t dstX = 0; dstX < m_NumTablesX2; ++dstX ) {
                intptr_t i = dstX % m_NumTablesX * m_NumCoefsX + iCoef;
                m_TablesX2[iCoef * m_NumTablesX2 + dstX] = m_TablesX[i];
            }
        }
        // Y coefs
        std::vector<float> tablesY(m_NumCoefsY);
        for ( intptr_t dstY = 0; dstY < m_NumTablesY; ++dstY ) {
            int16_t * table = &m_TablesY[dstY * m_NumCoefsY];
            double sumCoefs = setLanczosTable(degree, m_SrcH, m_DstH, dstY, pxScale, m_NumCoefsY, &tablesY[0]);
            adjustCoefs(&tablesY[0], &tablesY[m_NumCoefsY], sumCoefs, &table[0]);
        }

        // allocate workspace
        m_Work.reserve(m_SrcW * m_DstH);
        m_Work.resize(m_SrcW * m_DstH);
        size_t maxW = std::max(m_SrcW, m_DstW);
        m_Nume.reserve(maxW);
        m_Nume.resize(maxW);
        m_Deno.reserve(maxW);
        m_Deno.resize(maxW);

        // calc indices
        m_IndicesX.reserve(m_DstW);
        m_IndicesX.resize(m_DstW);
        for ( intptr_t dstX = 0; dstX < m_DstW; ++dstX ) {
            uint16_t srcOX = dstX * m_SrcW / m_DstW + 1;
            m_IndicesX[dstX] = srcOX;
        }
    }

    //! dst[i] = src[i] * kBias / srcSum (src will be broken)
    void LanczosResizer::Impl::adjustCoefs(
        float * __restrict srcBegin, const float * srcEnd,
        float srcSum,
        int16_t * __restrict dst)
    {
        size_t numCoefs = srcEnd - srcBegin;
        int16_t dstSum = 0;

        for ( size_t i = 0; i < numCoefs; ++i ) {
            dst[i] = round(srcBegin[i] * kBias / srcSum);
            dstSum += dst[i];
        }
        while ( dstSum < kFixed1_0 ) {
            size_t i = std::distance(&srcBegin[0], std::max_element(&srcBegin[0], &srcBegin[m_NumCoefsX]));
            dst[i]++;
            srcBegin[i] = 0;
            dstSum++;
        }
        while ( dstSum > kFixed1_0 ) {
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
            for ( intptr_t y = 0; y < m_SrcH; ++y ) {
                for ( intptr_t x = 0; x < m_SrcW; ++x ) {
                    m_Work[m_SrcW * y + x] = src[srcSt * y + x];
                }
            }
        } else {
            // vertical
            intptr_t numCoefsOn2 = m_NumCoefsY / 2;
            // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstH / double(m_SrcH))
            intptr_t mainBegin = ((numCoefsOn2 - 1) * m_DstH + m_SrcH-1) / m_SrcH;
            intptr_t mainEnd = std::max<intptr_t>(0, (m_SrcH - numCoefsOn2) * m_DstH / m_SrcH);
            const int16_t * tablesY = &m_TablesY[0];
            intptr_t tableSize = m_NumTablesY * m_NumCoefsY;
            intptr_t iTable = 0;
            LinearIterator iSrcOY(m_DstH, m_SrcH);

            for ( intptr_t dstY = 0; dstY < mainBegin; ++dstY ) {
                //       srcOY = floor(dstY / scale) + 1
                intptr_t srcOY = *iSrcOY++ + 1;
                //            coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
                const int16_t * coefs = &tablesY[iTable];
                iTable += m_NumCoefsY;
                if ( iTable == tableSize ) {
                    iTable = 0;
                }
                resizeYborder(
                    srcSt, &src[0],
                    m_SrcW, &m_Work[m_SrcW * dstY],
                    srcOY,
                    coefs,
                    &m_Nume[0], &m_Deno[0]);
            }
            for ( intptr_t dstY = mainBegin; dstY < mainEnd; ++dstY ) {
                //       srcOY = floor(dstY / scale) + 1
                intptr_t srcOY = *iSrcOY++ + 1;
                //            coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
                const int16_t * coefs = &tablesY[iTable];
                iTable += m_NumCoefsY;
                if ( iTable == tableSize ) {
                    iTable = 0;
                }
                resizeYmain(
                    srcSt, &src[0],
                    m_SrcW, &m_Work[m_SrcW * dstY],
                    srcOY,
                    coefs);
            }
            for ( intptr_t dstY = mainEnd; dstY < m_DstH; ++dstY ) {
                //       srcOY = floor(dstY / scale) + 1
                intptr_t srcOY = *iSrcOY++ + 1;
                //            coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
                const int16_t * coefs = &tablesY[iTable];
                iTable += m_NumCoefsY;
                if ( iTable == tableSize ) {
                    iTable = 0;
                }
                resizeYborder(
                    srcSt, &src[0],
                    m_SrcW, &m_Work[m_SrcW * dstY],
                    srcOY,
                    coefs,
                    &m_Nume[0], &m_Deno[0]);
            }
        }
        for ( intptr_t y = 0; y < m_DstH; ++y ) {
            // horizontal
            resizeX(&m_Work[m_SrcW * y], &dst[dstSt * y]);
        }
    }

    void LanczosResizer::Impl::resizeYborder(
        intptr_t srcSt, const uint8_t * src,
        intptr_t dstW, int16_t * __restrict dst,
        intptr_t srcOY,
        const int16_t * coefs,
        int16_t * __restrict nume, int16_t * __restrict deno)
    {
        intptr_t numCoefsOn2 = m_NumCoefsY / 2;

        std::memset(nume, 0, dstW * sizeof(*nume));
        std::memset(deno, 0, dstW * sizeof(*deno));

        for ( intptr_t i = 0; i < m_NumCoefsY; ++i ) {
            int16_t coef = coefs[i];
            for ( intptr_t dstX = 0; dstX < dstW; ++dstX ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                if ( 0 <= srcY && srcY < m_SrcH ) {
                    nume[dstX] += src[dstX + srcSt * srcY] * coef;
                    deno[dstX] += coef;
                }
            }
        }
        for ( intptr_t dstX = 0; dstX < dstW; ++dstX ) {
            dst[dstX] = roundedDiv(nume[dstX], deno[dstX]);
        }
    }

    void LanczosResizer::Impl::resizeYmain(
        intptr_t srcSt, const uint8_t * src,
        intptr_t dstW, int16_t * __restrict dst,
        intptr_t srcOY,
        const int16_t * coefs)
    {
        intptr_t numCoefsOn2 = m_NumCoefsY / 2;
        intptr_t vecLen = alignFloor(dstW, kVecStep);

        for ( intptr_t dstX = 0; dstX < vecLen; dstX += kVecStep ) {
            __m128i s16x8Nume0 = _mm_setzero_si128();
            __m128i s16x8Nume1 = _mm_setzero_si128();
            for ( intptr_t i = 0; i < m_NumCoefsY; ++i ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                __m128i s16x8Coef = _mm_set1_epi16(coefs[i]);
                // nume[dstX] += src[dstX + srcSt*srcY] * coef;
                __m128i u8x16Src    = _mm_loadu_si128((const __m128i *)&src[dstX + srcSt*srcY]);
                __m128i s16x8Src0   = _mm_cvtepu8_epi16(u8x16Src);
                __m128i s16x8Src1   = _mm_unpackhi_epi8(u8x16Src, _mm_setzero_si128());
                __m128i s16x8iNume0 = _mm_mullo_epi16(s16x8Src0, s16x8Coef);
                __m128i s16x8iNume1 = _mm_mullo_epi16(s16x8Src1, s16x8Coef);
                s16x8Nume0 = _mm_add_epi16(s16x8Nume0, s16x8iNume0);
                s16x8Nume1 = _mm_add_epi16(s16x8Nume1, s16x8iNume1);
            }

            // dst[dstX] = clamp<int16_t>(0, 255, roundedDiv(nume[dstX], kBias));
            __m128i s16x8Dst0 = cvtFixedToInt(s16x8Nume0);
            __m128i s16x8Dst1 = cvtFixedToInt(s16x8Nume1);
            _mm_storeu_si128((__m128i*)&dst[dstX + 0], s16x8Dst0);
            _mm_storeu_si128((__m128i*)&dst[dstX + 8], s16x8Dst1);
        }

        for ( intptr_t dstX = vecLen; dstX < dstW; dstX++ ) {
            int16_t nume = 0;
            for ( intptr_t i = 0; i < m_NumCoefsY; ++i ) {
                int16_t  coef = coefs[i];
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                nume += src[dstX + srcSt * srcY] * coef;
            }
            dst[dstX] = roundedDiv(nume, kBias);
        }
    }

    void LanczosResizer::Impl::resizeX(const int16_t * src, uint8_t * __restrict dst)
    {
        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstW / double(m_SrcW))
        intptr_t mainBegin = alignCeil(((numCoefsOn2 - 1) * m_DstW + m_SrcW-1) / m_SrcW, kVecStep);
        intptr_t mainEnd = std::max<intptr_t>(0, (m_SrcW - numCoefsOn2) * m_DstW / m_SrcW);
        intptr_t mainLen = alignFloor(mainEnd - mainBegin, kVecStep);
        mainEnd = mainBegin + mainLen;

        resizeXborder(src, dst, 0,         mainBegin);
        resizeXmain  (src, dst, mainBegin, mainEnd);
        resizeXborder(src, dst, mainEnd,   m_DstW);
    }

    void LanczosResizer::Impl::resizeXborder(
        const int16_t * src, uint8_t * __restrict dst,
        intptr_t begin, intptr_t end)
    {
        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        const int16_t * tablesX2 = &m_TablesX2[0];
        const uint16_t * indices = &m_IndicesX[0];

        intptr_t coefBegin = begin % m_NumTablesX2;
        intptr_t iCoef = coefBegin;
        for ( intptr_t dstX = begin; dstX < end; ++dstX ) {
            //       srcOX = floor(dstX / scale) + 1;
            intptr_t srcOX = indices[dstX];
            int16_t  nume  = 0;
            int16_t  deno  = 0;

            for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
                const int16_t * coefs = &tablesX2[i * m_NumTablesX2];
                intptr_t srcX = srcOX - numCoefsOn2 + i;
                if ( 0 <= srcX && srcX < m_SrcW ) {
                    int16_t coef = coefs[iCoef];
                    nume += src[srcX] * coef;
                    deno += coef;
                }
            }

            // iCoef = dstX % m_NumTablesX2;
            iCoef++;
            if ( iCoef == m_NumTablesX2 ) {
                iCoef = 0;
            }

            dst[dstX] = clamp<int16_t>(0, 255, roundedDiv(nume, deno));
        }
    }

    void LanczosResizer::Impl::resizeXmain(
        const int16_t * src, uint8_t * __restrict dst,
        intptr_t begin, intptr_t end)
    {
        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        const int16_t * tablesX2 = &m_TablesX2[0];
        const uint16_t * indices = &m_IndicesX[0];

        intptr_t coefBegin = begin % m_NumTablesX2;
        intptr_t iCoef = coefBegin;
        for ( intptr_t dstX = begin; dstX < end; dstX += kVecStep ) {
            // nume             = 0;
            __m128i s16x8Nume0  = _mm_setzero_si128();
            __m128i s16x8Nume1  = _mm_setzero_si128();
            // srcOX            = floor(dstX / scale) + 1;
            __m128i u16x8SrcOX0 = _mm_loadu_si128((const __m128i*)&indices[dstX + 0]);
            __m128i u16x8SrcOX1 = _mm_loadu_si128((const __m128i*)&indices[dstX + 8]);

            for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
                const int16_t * coefs = &tablesX2[i * m_NumTablesX2];

                // srcX             = srcOX - numCoefsOn2 + i;
                __m128i s16x8Offset = _mm_set1_epi16(i - numCoefsOn2);
                __m128i u16x8SrcX0  = _mm_add_epi16(u16x8SrcOX0, s16x8Offset);
                __m128i u16x8SrcX1  = _mm_add_epi16(u16x8SrcOX1, s16x8Offset);
                // nume            += src[srcX] * coefs[iCoef];
                __m128i s16x8Src0   = gather_epi16((const int16_t *)src, u16x8SrcX0);
                __m128i s16x8Src1   = gather_epi16((const int16_t *)src, u16x8SrcX1);
                __m128i s16x8Coefs  = _mm_loadu_si128((const __m128i *)&coefs[iCoef]);
                __m128i s16x8iNume0 = _mm_mullo_epi16(s16x8Src0, s16x8Coefs);
                __m128i s16x8iNume1 = _mm_mullo_epi16(s16x8Src1, s16x8Coefs);

                s16x8Nume0 = _mm_add_epi16(s16x8Nume0, s16x8iNume0);
                s16x8Nume1 = _mm_add_epi16(s16x8Nume1, s16x8iNume1);
            }

            // dst[dstX] = clamp<int16_t>(0, 255, roundedDiv(nume, kBias));
            __m128i s16x8Dst0 = cvtFixedToInt(s16x8Nume0);
            __m128i s16x8Dst1 = cvtFixedToInt(s16x8Nume1);
            __m128i u8x16Dst  = _mm_packus_epi16(s16x8Dst0, s16x8Dst1);
            _mm_storeu_si128((__m128i*)&dst[dstX], u8x16Dst);

            // iCoef = dstX % m_NumTablesX2;
            iCoef += kVecStep;
            if ( iCoef == m_NumTablesX2 ) {
                iCoef = 0;
            }
        }
    }

}
