#include <stdint.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <immintrin.h>

#include "libiqo/IQOLanczosResizer.hpp"
#include <cstdio>

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
        void resizeYborder(
            intptr_t srcSt, const uint8_t * src,
            intptr_t dstSt, intptr_t dstW, float * dst,
            intptr_t srcOY, intptr_t dstY,
            const float * coefs,
            float * deno);
        void resizeYmain(
            intptr_t srcSt, const uint8_t * src,
            intptr_t dstSt, intptr_t dstW, float * dst,
            intptr_t srcOY, intptr_t dstY,
            const float * coefs);
        void resizeX(const float * src, uint8_t * dst, float * nume, float * deno);

        intptr_t m_SrcW;
        intptr_t m_SrcH;
        intptr_t m_DstW;
        intptr_t m_DstH;
        intptr_t m_NumCoefsX;
        intptr_t m_NumCoefsY;
        intptr_t m_NumTablesX;
        intptr_t m_NumTablesY;
        std::vector<float> m_TablesX;   //!< Lanczos table * m_NumTablesX (interleaved)
        std::vector<float> m_TablesY;   //!< Lanczos table * m_NumTablesY
        std::vector<int32_t> m_IndicesX;
        std::vector<float> m_Work;
        std::vector<float> m_Nume;
        std::vector<float> m_Deno;
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
        m_TablesX.reserve(m_DstW * m_NumCoefsX);
        m_TablesX.resize(m_DstW * m_NumCoefsX);
        m_TablesY.reserve(m_NumCoefsY * m_NumTablesY);
        m_TablesY.resize(m_NumCoefsY * m_NumTablesY);
        m_IndicesX.reserve(m_DstW);
        m_IndicesX.resize(m_DstW);

        std::vector<float> tablesX(m_NumCoefsX * m_NumTablesX);
        for ( intptr_t dstX = 0; dstX < m_NumTablesX; ++dstX ) {
            float * table = &tablesX[dstX * m_NumCoefsX];
            double sumCoefs = setLanczosTable(degree, m_SrcW, m_DstW, dstX, pxScale, m_NumCoefsX, table);
            for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
                table[i] /= sumCoefs;
            }
        }
        // interleave for vectorization
        //
        // from: A0A1A2
        //       B0B1B2
        //
        //   to: A0 B0 .. A0 B0
        //       A1 B1 .. A1 B1
        //       A2 B2 .. A2 B2
        for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
            for ( intptr_t dstX = 0; dstX < m_DstW; ++dstX ) {
                m_TablesX[dstX + i * m_DstW] = tablesX[i + dstX % m_NumTablesX * m_NumCoefsX];
            }
        }
        for ( intptr_t dstX = 0; dstX < m_DstW; ++dstX ) {
            //       srcOX = floor(dstX / scale)
            intptr_t srcOX = dstX * m_SrcW / m_DstW;
            m_IndicesX[dstX] = srcOX;
        }

        for ( intptr_t dstY = 0; dstY < m_NumTablesY; ++dstY ) {
            float * table = &m_TablesY[dstY * m_NumCoefsY];
            double sumCoefs = setLanczosTable(degree, m_SrcH, m_DstH, dstY, pxScale, m_NumCoefsY, table);
            for ( intptr_t i = 0; i < m_NumCoefsY; ++i ) {
                table[i] /= sumCoefs;
            }
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

    void LanczosResizer::Impl::resize(size_t srcSt, const uint8_t * src, size_t dstSt, uint8_t * dst)
    {
        float * tmp = &m_Work[0];
        float * nume = &m_Nume[0];
        float * deno = &m_Deno[0];

        // resize
        if ( m_SrcH == m_DstH ) {
            for ( intptr_t y = 0; y < m_SrcH; ++y ) {
                for ( intptr_t x = 0; x < m_SrcW; ++x ) {
                    tmp[srcSt * y + x] = src[srcSt * y + x];
                }
            }
        } else {
            // vertical
            intptr_t numCoefsOn2 = m_NumCoefsY / 2;
            // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstH / double(m_SrcH))
            intptr_t mainBegin = ((numCoefsOn2 - 1) * m_DstH + m_SrcH-1) / m_SrcH;
            intptr_t mainEnd = std::max<intptr_t>(0, (m_SrcH - numCoefsOn2) * m_DstH / m_SrcH);
            const float * tablesY = &m_TablesY[0];
            intptr_t tableSize = m_NumTablesY * m_NumCoefsY;
            intptr_t iTable = 0;
            LinearIterator iSrcOY(m_DstH, m_SrcH);

            for ( intptr_t dstY = 0; dstY < mainBegin; ++dstY ) {
                //       srcOY = floor(dstY / scale) + 1
                intptr_t srcOY = *iSrcOY++ + 1;
                //            coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
                const float * coefs = &tablesY[iTable];
                iTable += m_NumCoefsY;
                if ( iTable == tableSize ) {
                    iTable = 0;
                }
                resizeYborder(
                    srcSt, &src[0],
                    m_SrcW, m_SrcW, &tmp[0],
                    srcOY, dstY,
                    coefs,
                    deno);
            }
            for ( intptr_t dstY = mainBegin; dstY < mainEnd; ++dstY ) {
                //       srcOY = floor(dstY / scale) + 1
                intptr_t srcOY = *iSrcOY++ + 1;
                //            coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
                const float * coefs = &tablesY[iTable];
                iTable += m_NumCoefsY;
                if ( iTable == tableSize ) {
                    iTable = 0;
                }
                resizeYmain(
                    srcSt, &src[0],
                    m_SrcW, m_SrcW, &tmp[0],
                    srcOY, dstY,
                    coefs);
            }
            for ( intptr_t dstY = mainEnd; dstY < m_DstH; ++dstY ) {
                //       srcOY = floor(dstY / scale) + 1
                intptr_t srcOY = *iSrcOY++ + 1;
                //            coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
                const float * coefs = &tablesY[iTable];
                iTable += m_NumCoefsY;
                if ( iTable == tableSize ) {
                    iTable = 0;
                }
                resizeYborder(
                    srcSt, &src[0],
                    m_SrcW, m_SrcW, &tmp[0],
                    srcOY, dstY,
                    coefs,
                    deno);
            }
        }
        // horizontal
        for ( intptr_t y = 0; y < m_DstH; ++y ) {
            resizeX(&tmp[srcSt * y], &dst[dstSt * y], nume, deno);
        }
    }

    void LanczosResizer::Impl::resizeYborder(
        intptr_t srcSt, const uint8_t * src,
        intptr_t dstSt, intptr_t dstW, float * dst,
        intptr_t srcOY, intptr_t dstY,
        const float * coefs,
        float * deno)
    {
        intptr_t numCoefsOn2 = m_NumCoefsY / 2;
        float * nume = &dst[dstSt * dstY];

        std::memset(nume, 0, dstW * sizeof(*nume));
        std::memset(deno, 0, dstW * sizeof(*deno));

        for ( intptr_t i = 0; i < m_NumCoefsY; ++i ) {
            float coef = coefs[i];
            for ( intptr_t dstX = 0; dstX < dstW; ++dstX ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                
                if ( 0 <= srcY && srcY < m_SrcH ) {
                    nume[dstX] += src[dstX + srcSt * srcY] * coef;
                    deno[dstX] += coef;
                }
            }
        }
        for ( intptr_t dstX = 0; dstX < dstW; ++dstX ) {
            nume[dstX] /= deno[dstX];
        }
    }

    void LanczosResizer::Impl::resizeYmain(
        intptr_t srcSt, const uint8_t * src,
        intptr_t dstSt, intptr_t dstW, float * dst,
        intptr_t srcOY, intptr_t dstY,
        const float * coefs)
    {
        intptr_t numCoefsOn2 = m_NumCoefsY / 2;
        float * nume = &dst[dstSt * dstY];

        std::memset(nume, 0, dstW * sizeof(*nume));

        for ( intptr_t i = 0; i < m_NumCoefsY; ++i ) {
            float coef = coefs[i];
            for ( intptr_t dstX = 0; dstX < dstW; ++dstX ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                nume[dstX] += src[dstX + srcSt * srcY] * coef;
            }
        }
    }

    void LanczosResizer::Impl::resizeX(const float * src, uint8_t * dst, float * nume, float * deno)
    {
        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        const float * tablesX = &m_TablesX[0];
        // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstW / double(m_SrcW))
        intptr_t mainBegin = ((numCoefsOn2 - 1) * m_DstW + m_SrcW-1) / m_SrcW;
        intptr_t mainEnd = std::max<intptr_t>(0, (m_SrcW - numCoefsOn2) * m_DstW / m_SrcW);
        intptr_t mainLen = (mainEnd - mainBegin) & intptr_t(-16);
        mainEnd = mainBegin + mainLen;
        LinearIterator iSrcOX(m_DstW, m_SrcW);
        int32_t * indices = &m_IndicesX[0];

        std::memset(nume, 0, m_DstW * sizeof(*nume));
        std::memset(deno, 0, m_DstW * sizeof(*deno));

        // before main
        for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
            int32_t offset = -numCoefsOn2 + 1 + i;
            const float * coefs = &tablesX[i * m_DstW];

            for ( intptr_t dstX = 0; dstX < mainBegin; ++dstX ) {
                //       srcX = floor(dstX / scale) - numCoefsOn2 + 1 + i;
                intptr_t srcX = indices[dstX] + offset;

                if ( 0 <= srcX && srcX < m_SrcW ) {
                    nume[dstX] += src[srcX] * coefs[dstX];
                    deno[dstX] += coefs[dstX];
                }
            }
        }
        for ( intptr_t dstX = 0; dstX < mainBegin; ++dstX ) {
            dst[dstX] = clamp<int>(0, 255, round(nume[dstX] / deno[dstX]));
        }

        // main
        for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
            int32_t offset = -numCoefsOn2 + 1 + i;
            const float * coefs = &tablesX[i * m_DstW];
            __m256i vOffset   = _mm256_set1_epi32(offset);

            for ( intptr_t dstX = mainBegin; dstX < mainEnd; dstX += 16 ) {
                //       srcX = floor(dstX / scale) - numCoefsOn2 + 1 + i;
                //intptr_t srcX = indices[dstX + j] + offset;
                __m256i vSrcX0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)&indices[dstX + 0]), vOffset);
                __m256i vSrcX8 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)&indices[dstX + 8]), vOffset);

                //nume[dstX + j] += src[srcX] * coefs[dstX + j];
                __m256  vSrc0     = _mm256_i32gather_ps(src, vSrcX0, sizeof(float));
                __m256  vSrc8     = _mm256_i32gather_ps(src, vSrcX8, sizeof(float));
                __m256  vCoefs0   = _mm256_loadu_ps(&coefs[dstX + 0]);
                __m256  vCoefs8   = _mm256_loadu_ps(&coefs[dstX + 8]);
                __m256  vNume0    = _mm256_loadu_ps(&nume[dstX + 0]);
                __m256  vNume8    = _mm256_loadu_ps(&nume[dstX + 8]);
                vNume0 = _mm256_fmadd_ps(vSrc0, vCoefs0, vNume0);
                vNume8 = _mm256_fmadd_ps(vSrc8, vCoefs8, vNume8);
                _mm256_storeu_ps(&nume[dstX + 0], vNume0);
                _mm256_storeu_ps(&nume[dstX + 8], vNume8);
            }
        }
        for ( intptr_t dstX = mainBegin; dstX < mainEnd; dstX += 16 ) {
            //dst[dstX] = clamp<int>(0, 255, round(nume[dstX]));
            __m128i vNumeShuffle = _mm_set_epi8(
                15, 14, 13, 12,   7,  6,  5,  4,
                11, 10,  9,  8,   3,  2,  1,  0);
            __m256 vNume0 = _mm256_loadu_ps(&nume[dstX + 0]);
            __m256 vNume8 = _mm256_loadu_ps(&nume[dstX + 8]);
            vNume0 = _mm256_round_ps(vNume0, _MM_FROUND_TO_NEAREST_INT);
            vNume8 = _mm256_round_ps(vNume8, _MM_FROUND_TO_NEAREST_INT);
            __m256i v256Nume0 = _mm256_cvttps_epi32(vNume0);                // uint32_t 7 6 5 4 3 2 1 0
            __m256i v256Nume8 = _mm256_cvttps_epi32(vNume8);                // uint32_t F E D C B A 9 8
            __m256i v256Nume  = _mm256_packus_epi32(v256Nume0, v256Nume8);  // uint16_t FEDC7654BA983210
            __m128i v128Nume0 = _mm256_extracti128_si256(v256Nume, 0);      // uint16_t BA983210
            __m128i v128Nume8 = _mm256_extracti128_si256(v256Nume, 1);      // uint16_t FEDC7654
            __m128i v128Nume  = _mm_packus_epi16(v128Nume0, v128Nume8);     // uint8_t  FEDC7654BA983210
            __m128i vNume = _mm_shuffle_epi8(v128Nume, vNumeShuffle);       // uint8_t  FEDCBA9876543210
            _mm_storeu_si128((__m128i*)&dst[dstX], vNume);
        }

        // after main
        for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
            int32_t offset = -numCoefsOn2 + 1 + i;
            const float * coefs = &tablesX[i * m_DstW];

            for ( intptr_t dstX = mainEnd; dstX < m_DstW; ++dstX ) {
                //       srcX = floor(dstX / scale) - numCoefsOn2 + 1 + i;
                intptr_t srcX = indices[dstX] + offset;

                if ( 0 <= srcX && srcX < m_SrcW ) {
                    nume[dstX] += src[srcX] * coefs[dstX];
                    deno[dstX] += coefs[dstX];
                }
            }
        }
        for ( intptr_t dstX = mainEnd; dstX < m_DstW; ++dstX ) {
            dst[dstX] = clamp<int>(0, 255, round(nume[dstX] / deno[dstX]));
        }
    }

}
