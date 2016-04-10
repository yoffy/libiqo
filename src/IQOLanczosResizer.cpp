#include <stdint.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

#include "libiqo/IQOLanczosResizer.hpp"


namespace {

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
        if ( x >= 0 ) {
            return std::floor(x + T(0.5));
        } else {
            return std::ceil(x - T(0.5));
        }
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

}

namespace iqo {

    class LanczosResizer::Impl
    {
    public:
        //! Constructor
        Impl(unsigned int degree, size_t srcW, size_t srcH, size_t dstW, size_t dstH, size_t pxScale);

        //! Run image resizing
        void resize(size_t srcSt, const uint8_t * src, size_t dstSt, uint8_t * dst);

    private:
        //! round(a / b)
        static int16_t roundedDiv(int16_t a, int16_t b)
        {
            return (a + kBias/2) / b;
        }

        //! dst[i] = src[i] * kBias / srcSum (src will be broken)
        void adjustCoefs(
            float * srcBegin, float * srcEnd,
            float srcSum,
            int16_t * dst);

        void resizeYborder(
            intptr_t srcSt, const uint8_t * src,
            intptr_t dstW, int16_t * dst,
            intptr_t srcOY,
            const int16_t * coefs,
            int16_t * nume, int16_t * deno);
        void resizeYmain(
            intptr_t srcSt, const uint8_t * src,
            intptr_t dstW, int16_t * dst,
            intptr_t srcOY,
            const int16_t * coefs,
            int16_t * nume);
        void resizeX(const int16_t * src, uint8_t * dst);

        static const int16_t kBias = 64; //! for fixed point
        intptr_t m_SrcW;
        intptr_t m_SrcH;
        intptr_t m_DstW;
        intptr_t m_DstH;
        intptr_t m_NumCoefsX;
        intptr_t m_NumCoefsY;
        intptr_t m_NumTablesX;
        intptr_t m_NumTablesY;
        std::vector<int16_t> m_TablesX;   //!< Lanczos table * m_NumTablesX
        std::vector<int16_t> m_TablesY;   //!< Lanczos table * m_NumTablesY
        std::vector<int16_t> m_Work;
        std::vector<int16_t> m_Nume;
        std::vector<int16_t> m_Deno;
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
        m_NumTablesY = m_DstH / gcd(m_SrcH, m_DstH);
        m_TablesX.reserve(m_NumCoefsX * m_NumTablesX);
        m_TablesX.resize(m_NumCoefsX * m_NumTablesX);
        m_TablesY.reserve(m_NumCoefsY * m_NumTablesY);
        m_TablesY.resize(m_NumCoefsY * m_NumTablesY);

        std::vector<float> tablesX(m_NumCoefsX);
        for ( intptr_t dstX = 0; dstX < m_NumTablesX; ++dstX ) {
            int16_t * table = &m_TablesX[dstX * m_NumCoefsX];
            double sumCoefs = setLanczosTable(degree, m_SrcW, m_DstW, dstX, pxScale, m_NumCoefsX, &tablesX[0]);
            adjustCoefs(&tablesX[0], &tablesX[m_NumCoefsX], sumCoefs, &table[0]);
        }
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
    }

    void LanczosResizer::Impl::adjustCoefs(
        float * srcBegin, float * srcEnd,
        float srcSum,
        int16_t * dst)
    {
        size_t numCoefs = srcEnd - srcBegin;
        int16_t dstSum = 0;

        for ( size_t i = 0; i < numCoefs; ++i ) {
            dst[i] = round(srcBegin[i] * kBias / srcSum);
            dstSum += dst[i];
        }
        while ( dstSum < kBias ) {
            size_t i = std::distance(&srcBegin[0], std::max_element(&srcBegin[0], &srcBegin[m_NumCoefsX]));
            dst[i]++;
            srcBegin[i] = 0;
            dstSum++;
        }
        while ( dstSum > kBias ) {
            size_t i = std::distance(&srcBegin[0], std::max_element(&srcBegin[0], &srcBegin[m_NumCoefsX]));
            dst[i]--;
            srcBegin[i] = 0;
            dstSum--;
        }
    }

    void LanczosResizer::Impl::resize(size_t srcSt, const uint8_t * src, size_t dstSt, uint8_t * dst)
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
                    coefs,
                    &m_Nume[0]);
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
        intptr_t dstW, int16_t * dst,
        intptr_t srcOY,
        const int16_t * coefs,
        int16_t * nume, int16_t * deno)
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
        intptr_t dstW, int16_t * dst,
        intptr_t srcOY,
        const int16_t * coefs,
        int16_t * nume)
    {
        intptr_t numCoefsOn2 = m_NumCoefsY / 2;

        std::memset(nume, 0, dstW * sizeof(*nume));

        for ( intptr_t i = 0; i < m_NumCoefsY; ++i ) {
            int16_t coef = coefs[i];
            for ( intptr_t dstX = 0; dstX < dstW; ++dstX ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                nume[dstX] += src[dstX + srcSt * srcY] * coef;
            }
        }

        for ( intptr_t dstX = 0; dstX < dstW; ++dstX ) {
            dst[dstX] = roundedDiv(nume[dstX], kBias);
        }
    }

    void LanczosResizer::Impl::resizeX(const int16_t * src, uint8_t * dst)
    {
        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        const int16_t * tablesX = &m_TablesX[0];
        // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstW / double(m_SrcW))
        intptr_t mainBegin = ((numCoefsOn2 - 1) * m_DstW + m_SrcW-1) / m_SrcW;
        intptr_t mainEnd = std::max<intptr_t>(0, (m_SrcW - numCoefsOn2) * m_DstW / m_SrcW);
        LinearIterator iSrcOX(m_DstW, m_SrcW);
        intptr_t tableSize = m_NumTablesX * m_NumCoefsX;
        intptr_t iTable = 0;

        // before main
        for ( intptr_t dstX = 0; dstX < mainBegin; ++dstX ) {
            //       srcOX = floor(dstX / scale) + 1;
            intptr_t srcOX = *iSrcOX++ + 1;
            //            coefs = &tablesX[dstX % m_NumTablesX * m_NumCoefsX];
            const int16_t * coefs = &tablesX[iTable];
            int16_t sum = 0;
            int16_t deno = 0;

            iTable += m_NumCoefsX;
            if ( iTable == tableSize ) {
                iTable = 0;
            }
            for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
                intptr_t srcX = srcOX - numCoefsOn2 + i;
                if ( 0 <= srcX && srcX < m_SrcW ) {
                    sum += src[srcX] * coefs[i];
                    deno += coefs[i];
                }
            }

            dst[dstX] = clamp<int16_t>(0, 255, roundedDiv(sum, deno));
        }

        for ( intptr_t dstX = mainBegin; dstX < mainEnd; ++dstX ) {
            //       srcOX = floor(dstX / scale) + 1;
            intptr_t srcOX = *iSrcOX++ + 1;
            //            coefs = &tablesX[dstX % m_NumTablesX * m_NumCoefsX];
            const int16_t * coefs = &tablesX[iTable];
            int sum = 0; // rather than int16_t for gcc 4.8 vectorization

            iTable += m_NumCoefsX;
            if ( iTable == tableSize ) {
                iTable = 0;
            }
            for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
                intptr_t srcX = srcOX - numCoefsOn2 + i;
                sum += src[srcX] * coefs[i];
            }

            dst[dstX] = clamp<int16_t>(0, 255, roundedDiv(sum, 64));
        }

        // after main
        for ( intptr_t dstX = mainEnd; dstX < m_DstW; ++dstX ) {
            //       srcOX = floor(dstX / scale) + 1;
            intptr_t srcOX = *iSrcOX++ + 1;
            //            coefs = &tablesX[dstX % m_NumTablesX * m_NumCoefsX];
            const int16_t * coefs = &tablesX[iTable];
            int16_t sum = 0;
            int16_t deno = 0;

            iTable += m_NumCoefsX;
            if ( iTable == tableSize ) {
                iTable = 0;
            }
            for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
                intptr_t srcX = srcOX - numCoefsOn2 + i;
                if ( 0 <= srcX && srcX < m_SrcW ) {
                    sum += src[srcX] * coefs[i];
                    deno += coefs[i];
                }
            }

            dst[dstX] = clamp<int16_t>(0, 255, roundedDiv(sum, deno));
        }
    }

}
