#include <stdint.h>
#include <cmath>
#include <cstring>
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
    //! Calculate Lanczos coefficients from `-degree * pxScale` to `+degree * pxScale`.
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
            //double beginX = (-degree - 0.5 + 0.5*scale)*pxScale + std::fmod((1 - srcOffset) * scale * pxScale, 1.0);

            //----- more accurate -----
            // srcOffset = std::fmod(dstOffset / scale, 1.0)
            //           = (dstOffset * srcLen % dstLen) / dstLen;

            // (-degree - 0.5 + 0.5*scale)*pxScale
            // = (-degree - 0.5 + 0.5*dstLen/srcLen) * pxScale
            // = (-degree - 0.5)*pxScale + 0.5*dstLen*pxScale/srcLen

            // std::fmod((1 - srcOffset) * scale * pxScale, 1.0)
            // = std::fmod((1 - (dstOffset * srcLen % dstLen)/dstLen) * (dstLen/srcLen) * pxScale, 1.0)
            // = std::fmod((dstLen/srcLen - (dstOffset * srcLen % dstLen)/dstLen*(dstLen/srcLen)) * pxScale, 1.0)
            // = std::fmod((dstLen/srcLen - (dstOffset * srcLen % dstLen)/srcLen) * pxScale, 1.0)
            // = std::fmod(((dstLen - (dstOffset * srcLen % dstLen))/srcLen) * pxScale, 1.0)
            // = std::fmod( (dstLen - (dstOffset * srcLen % dstLen))         * pxScale, srcLen) / srcLen
            // = ((dstLen - dstOffset*srcLen%dstLen) * pxScale % srcLen) / srcLen
            beginX =
                (-degree - 0.5)*pxScale + 0.5*dstLen*pxScale/srcLen
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
        void resizeX(const uint8_t * src, float * dst);
        void resizeYside(const float * src, intptr_t dstY, intptr_t dstSt, uint8_t * dst, float * nume, float * deno);
        void resizeYmain(const float * src, intptr_t dstY, intptr_t dstSt, uint8_t * dst, float * nume);

        intptr_t m_SrcW;
        intptr_t m_SrcH;
        intptr_t m_DstW;
        intptr_t m_DstH;
        intptr_t m_NumCoefsX;
        intptr_t m_NumCoefsY;
        intptr_t m_NumTablesX;
        intptr_t m_NumTablesY;
        std::vector<float> m_TablesX;   //!< Lanczos table * m_NumTablesX
        std::vector<float> m_SumsX;     //!< Sum of Lanczos table * m_NumTablesX
        std::vector<float> m_TablesY;   //!< Lanczos table * m_NumTablesY
        std::vector<float> m_SumsY;     //!< Sum of Lanczos table * m_NumTablesY
        std::vector<float> m_Work;
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
            m_NumCoefsX = 2 * intptr_t(std::ceil((degree * m_SrcW) / double(m_DstW)));
        }
        if ( m_SrcH <= m_DstH ) {
            // vertical: up-sampling
            m_NumCoefsY = 2 * degree;
        } else {
            // vertical: down-sampling
            // n = 2*degree / scale
            m_NumCoefsY = 2 * intptr_t(std::ceil((degree * m_SrcH) / double(m_DstH)));
        }
        m_NumTablesX = m_DstW / gcd(m_SrcW, m_DstW);
        m_NumTablesY = m_DstH / gcd(m_SrcH, m_DstH);
        m_TablesX.reserve(m_NumCoefsX * m_NumTablesX);
        m_TablesX.resize(m_NumCoefsX * m_NumTablesX);
        m_SumsX.reserve(m_NumTablesX);
        m_SumsX.resize(m_NumTablesX);
        m_TablesY.reserve(m_NumCoefsY * m_NumTablesY);
        m_TablesY.resize(m_NumCoefsY * m_NumTablesY);
        m_SumsY.reserve(m_NumTablesY);
        m_SumsY.resize(m_NumTablesY);

        for ( intptr_t dstX = 0; dstX < m_NumTablesX; ++dstX ) {
            float * table = &m_TablesX[dstX * m_NumCoefsX];
            m_SumsX[dstX] = setLanczosTable(degree, m_SrcW, m_DstW, dstX, pxScale, m_NumCoefsX, table);
        }
        for ( intptr_t dstY = 0; dstY < m_NumTablesY; ++dstY ) {
            float * table = &m_TablesY[dstY * m_NumCoefsY];
            m_SumsY[dstY] = setLanczosTable(degree, m_SrcH, m_DstH, dstY, pxScale, m_NumCoefsY, table);
        }
    }

    void LanczosResizer::Impl::resize(size_t srcSt, const uint8_t * src, size_t dstSt, uint8_t * dst)
    {
        // allocate workspace
        m_Work.reserve(dstSt * (m_SrcH + 2));
        m_Work.resize(dstSt * (m_SrcH + 2));
        float * tmp = &m_Work[0];
        float * tmpNume = &m_Work[dstSt * m_SrcH];
        float * tmpDeno = &tmpNume[dstSt];

        // resize
        for ( intptr_t y = 0; y < m_SrcH; ++y ) {
            // horizontal
            resizeX(&src[srcSt * y], &tmp[dstSt * y]);
        }
        if ( m_SrcH != m_DstH ) {
            // vertical
            intptr_t numCoefsOn2 = m_NumCoefsY / 2;
            // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstH / double(m_SrcH))
            intptr_t mainBegin = ((numCoefsOn2 - 1) * m_DstH + m_SrcH-1) / m_SrcH;
            intptr_t mainEnd = std::max<intptr_t>(0, (m_SrcH - numCoefsOn2) * m_DstH / m_SrcH);

            for ( intptr_t dstY = 0; dstY < mainBegin; ++dstY ) {
                resizeYside(&tmp[0], dstY, dstSt, &dst[0], tmpNume, tmpDeno);
            }
            for ( intptr_t dstY = mainBegin; dstY < mainEnd; ++dstY ) {
                resizeYmain(&tmp[0], dstY, dstSt, &dst[0], tmpNume);
            }
            for ( intptr_t dstY = mainEnd; dstY < m_DstH; ++dstY ) {
                resizeYside(&tmp[0], dstY, dstSt, &dst[0], tmpNume, tmpDeno);
            }
        }
    }

    void LanczosResizer::Impl::resizeX(const uint8_t * src, float * dst)
    {
        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        const float * tablesX = &m_TablesX[0];
        const float * sumCoefs = &m_SumsX[0];
        // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstW / double(m_SrcW))
        intptr_t mainBegin = ((numCoefsOn2 - 1) * m_DstW + m_SrcW-1) / m_SrcW;
        intptr_t mainEnd = std::max<intptr_t>(0, (m_SrcW - numCoefsOn2) * m_DstW / m_SrcW);

        // before main
        for ( intptr_t dstX = 0; dstX < mainBegin; ++dstX ) {
            //       srcOX = floor(dstX / scale)
            intptr_t srcOX = dstX * m_SrcW / m_DstW + 1;
            intptr_t iTable = dstX % m_NumTablesX;
            const float * coefs = &tablesX[iTable * m_NumCoefsX];
            float sum = 0;
            float deno = 0;

            for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
                intptr_t srcX = srcOX - numCoefsOn2 + i;
                if ( 0 <= srcX && srcX < m_SrcW ) {
                    sum += src[srcX] * coefs[i];
                    deno += coefs[i];
                }
            }

            dst[dstX] = sum / deno;
        }

        for ( intptr_t dstX = mainBegin; dstX < mainEnd; ++dstX ) {
            //       srcOX = floor(dstX / scale)
            intptr_t srcOX = dstX * m_SrcW / m_DstW + 1;
            intptr_t iTable = dstX % m_NumTablesX;
            const float * coefs = &tablesX[iTable * m_NumCoefsX];
            float sum = 0;

            for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
                intptr_t srcX = srcOX - numCoefsOn2 + i;
                sum += src[srcX] * coefs[i];
            }

            dst[dstX] = sum / sumCoefs[iTable];
        }

        // after main
        for ( intptr_t dstX = mainEnd; dstX < m_DstW; ++dstX ) {
            //       srcOX = floor(dstX / scale)
            intptr_t srcOX = dstX * m_SrcW / m_DstW + 1;
            intptr_t iTable = dstX % m_NumTablesX;
            const float * coefs = &tablesX[iTable * m_NumCoefsX];
            float sum = 0;
            float deno = 0;

            for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
                intptr_t srcX = srcOX - numCoefsOn2 + i;
                if ( 0 <= srcX && srcX < m_SrcW ) {
                    sum += src[srcX] * coefs[i];
                    deno += coefs[i];
                }
            }

            dst[dstX] = sum / deno;
        }
    }

    void LanczosResizer::Impl::resizeYside(const float * src, intptr_t dstY, intptr_t dstSt, uint8_t * dst, float * nume, float * deno)
    {
        intptr_t numCoefsOn2 = m_NumCoefsY / 2 - 1;
        const float * tablesY = &m_TablesY[0];
        intptr_t tail = m_SrcH - 1;
        //       srcOY = floor(dstY / scale)
        intptr_t srcOY = dstY * m_SrcH / m_DstH;
        intptr_t iTable = dstY % m_NumTablesY;
        const float * coefs = &tablesY[iTable * m_NumCoefsY];

        std::memset(nume, 0, m_DstW * sizeof(*nume));
        std::memset(deno, 0, m_DstW * sizeof(*deno));

        for ( intptr_t i = 0; i < m_NumCoefsY; ++i ) {
            float coef = coefs[i];
            for ( intptr_t dstX = 0; dstX < m_DstW; ++dstX ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                nume[dstX] += src[dstX + dstSt * clamp<intptr_t>(0, tail, srcY)] * coef;
                deno[dstX] += coef;
            }
        }
        for ( intptr_t dstX = 0; dstX < m_DstW; ++dstX ) {
            nume[dstX] = round(nume[dstX] / deno[dstX]);
        }
        for ( intptr_t dstX = 0; dstX < m_DstW; ++dstX ) {
            dst[dstX + dstSt * dstY] = std::max(0, std::min(255, int(nume[dstX])));
        }
    }

    void LanczosResizer::Impl::resizeYmain(const float * src, intptr_t dstY, intptr_t dstSt, uint8_t * dst, float * sum)
    {
        intptr_t numCoefsOn2 = m_NumCoefsY / 2 - 1;
        const float * tablesY = &m_TablesY[0];
        const float * sumCoefs = &m_SumsY[0];
        //       srcOY = floor(dstY / scale)
        intptr_t srcOY = dstY * m_SrcH / m_DstH;
        intptr_t iTable = dstY % m_NumTablesY;
        const float * coefs = &tablesY[iTable * m_NumCoefsY];
        float sumCoef = sumCoefs[iTable];

        std::memset(sum, 0, m_DstW * sizeof(*sum));

        for ( intptr_t i = 0; i < m_NumCoefsY; ++i ) {
            float coef = coefs[i];
            for ( intptr_t dstX = 0; dstX < m_DstW; ++dstX ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                sum[dstX] += src[dstX + dstSt * srcY] * coef;
            }
        }
        for ( intptr_t dstX = 0; dstX < m_DstW; ++dstX ) {
            sum[dstX] = round(sum[dstX] / sumCoef);
        }
        for ( intptr_t dstX = 0; dstX < m_DstW; ++dstX ) {
            dst[dstX + dstSt * dstY] = std::max(0, std::min(255, int(sum[dstX])));
        }
    }

}
