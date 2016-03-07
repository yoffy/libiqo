#include <stdint.h>
#include <cmath>
#include <vector>

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
    //! @param scale      Resize scale
    //! @param srcOffset  Offset of the origin (ex. 0.5)
    //! @param n          Size of table
    //! @param fTable     The table
    //! @return Sum of the table
    //!
    //! Calculate Lanczos coefficients from `-degree + srcOffset` to `+degree + srcOffset`.
    //!
    //! n should be `2*degree` when up sampling.
    //!
    //! @b Example:
    //! @code
    //! void func()
    //! {
    //!     float t[4];
    //!
    //!     setLanczosTable(2, 4, t, 0.5f, 1);
    //!
    //!     printf("%f, %f, %f, %f\n", t[0], t[1], t[2], t[3]);
    //! }
    //! @endcode
    //!
    //! @b Output:
    //! @code
    //! -0.063684, 0.573159, 0.573159, -0.063684
    //! @endcode
    template<typename T>
    T setLanczosTable(int degree, double scale, double srcOffset, int n, T * fTable)
    {
        scale = (scale < 1) ? scale : 1;

        // X is offset of Lanczos from center.
        // It will be source coordinate when up-sampling,
        // or, destination coordinate when down-sampling.
        double beginX = -degree + std::fmod((1 - srcOffset) * scale, 1.0);
        T fSum = 0;

        for ( intptr_t i = 0; i < n; ++i ) {
            double x = beginX + i * scale;
            T v = T(lanczos(degree, x));
            fTable[i] = v;
            fSum     += v;
        }

        return fSum;
    }

    template<typename T>
    T round(T x)
    {
        return x + T(0.5);
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
        //! @brief Constructor
        //! @param degree  The degree of Lanczos (ex. A=2 means Lanczos2)
        //! @param srcW    Width of source image
        //! @param srcH    Height of source image
        //! @param dstW    Width of destination image
        //! @param dstH    Height of destination image
        Impl(unsigned int degree, size_t srcW, size_t srcH, size_t dstW, size_t dstH);

        //! @brief Run image resizing
        //! @param srcSt  Stride of src (in byte)
        //! @param src    Source image
        //! @param dstSt  Stride of dst (in byte)
        //! @param dst    Destination image
        //!
        //! Size of src and dst has to be set in constructor.
        //!
        //! srcSt and dstSt are line length in byte.
        void resize(size_t srcSt, const uint8_t * src, size_t dstSt, uint8_t * dst);

    private:
        void resizeX(const uint8_t * src, uint8_t * dst);
        void resizeY(const uint8_t * src, intptr_t dstSt, uint8_t * dst);

        intptr_t m_SrcW;
        intptr_t m_SrcH;
        intptr_t m_DstW;
        intptr_t m_DstH;
        double m_ScaleX;
        double m_ScaleY;
        intptr_t m_NumCoefsX;
        intptr_t m_NumCoefsY;
        intptr_t m_NumTablesX;
        intptr_t m_NumTablesY;
        std::vector<float> m_TablesX;    //!< Lanczos table * m_NumTablesX
        std::vector<float> m_SumsX;     //!< Sum of Lanczos table * m_NumTablesX
        std::vector<float> m_TablesY;    //!< Lanczos table * m_NumTablesY
        std::vector<float> m_SumsY;     //!< Sum of Lanczos table * m_NumTablesY
        std::vector<uint8_t> m_Work;
        uint8_t * m_Tmp;    //!< alias of m_Work or dst
    };

    LanczosResizer::LanczosResizer(unsigned int degree, size_t srcW, size_t srcH, size_t dstW, size_t dstH)
    {
        m_Impl = new Impl(degree, srcW, srcH, dstW, dstH);
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
    LanczosResizer::Impl::Impl(unsigned int degree, size_t srcW, size_t srcH, size_t dstW, size_t dstH)
    {
        m_SrcW = srcW;
        m_SrcH = srcH;
        m_DstW = dstW;
        m_DstH = dstH;

        // scale
        m_ScaleX = double(m_DstW) / m_SrcW;
        m_ScaleY = double(m_DstH) / m_SrcH;

        // setup coefficients
        if ( m_SrcW <= m_DstW ) {
            m_NumCoefsX = 2 * degree;
        } else {
            m_NumCoefsX = intptr_t(std::ceil(2 * degree / m_ScaleX));
        }
        if ( m_SrcH <= m_DstH ) {
            m_NumCoefsY = 2 * degree;
        } else {
            m_NumCoefsY = intptr_t(std::ceil(2 * degree / m_ScaleY));
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

        for ( intptr_t x = 0; x < m_NumTablesX; ++x ) {
            double offset = std::fmod(x / m_ScaleX, 1.0);
            m_SumsX[x] = setLanczosTable(degree, m_ScaleX, offset, m_NumCoefsX, &m_TablesX[x * m_NumCoefsX]);
        }
        for ( intptr_t y = 0; y < m_NumTablesY; ++y ) {
            double offset = std::fmod(y / m_ScaleY, 1.0);
            m_SumsY[y] = setLanczosTable(degree, m_ScaleY, offset, m_NumCoefsY, &m_TablesY[y * m_NumCoefsY]);
        }
    }

    void LanczosResizer::Impl::resize(size_t srcSt, const uint8_t * src, size_t dstSt, uint8_t * dst)
    {
        // setup workspace
        bool isDstEnough = (m_SrcH == m_DstH) || (m_SrcH + m_NumCoefsY / 2 <= m_DstH);
        bool isAliased = (src < dst && dst < &src[srcSt * m_SrcH]) || (dst < src && src < &dst[dstSt * m_DstH]);
        if ( isDstEnough && ! isAliased ) {
            // reuse bottom of dst for workspace
            m_Work.resize(0);
            m_Tmp = &dst[dstSt * (m_DstH - m_SrcH)];
        } else {
            // allocate workspace
            m_Work.reserve(dstSt * m_SrcH);
            m_Work.resize(dstSt * m_SrcH);
            m_Tmp = &m_Work[0];
        }

        // resize
        for ( intptr_t y = 0; y < m_SrcH; ++y ) {
            resizeX(&src[srcSt * y], &m_Tmp[dstSt * y]);
        }
        if ( m_SrcH != m_DstH ) {
            //! @todo col major
            for ( intptr_t x = 0; x < m_DstW; ++x ) {
                resizeY(&m_Tmp[x], dstSt, &dst[x]);
            }
        }
    }

    void LanczosResizer::Impl::resizeX(const uint8_t * src, uint8_t * dst)
    {
        double invScale = 1.0 / m_ScaleX;
        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        const float * tablesX = &m_TablesX[0];
        const float * sumCoefs = &m_SumsX[0];
        intptr_t tail = m_SrcW - 1;

        for ( intptr_t dstX = 0; dstX < m_DstW; ++dstX ) {
            intptr_t srcOX = intptr_t(dstX * invScale);
            intptr_t iTable = dstX % m_NumTablesX;
            const float * coefs = &tablesX[iTable * m_NumCoefsX];
            float sum = 0;

            for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
                intptr_t srcX = srcOX - numCoefsOn2 + i;
                sum += src[clamp<intptr_t>(0, tail, srcX)] * coefs[i];
            }

            dst[dstX] = std::max(0, std::min(255, int(round(sum / sumCoefs[iTable]))));
        }
    }

    void LanczosResizer::Impl::resizeY(const uint8_t * src, intptr_t dstSt, uint8_t * dst)
    {
        double invScale = 1.0 / m_ScaleY;
        intptr_t numCoefsOn2 = m_NumCoefsY / 2;
        const float * tablesY = &m_TablesY[0];
        const float * sumCoefs = &m_SumsY[0];
        intptr_t tail = m_SrcH - 1;

        for ( intptr_t dstY = 0; dstY < m_DstH; ++dstY ) {
            intptr_t srcOY = intptr_t(dstY * invScale);
            intptr_t iTable = dstY % m_NumTablesX;
            const float * coefs = &tablesY[iTable * m_NumCoefsY];
            float sum = 0;

            for ( intptr_t i = 0; i < m_NumCoefsY; ++i ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                sum += src[dstSt * clamp<intptr_t>(0, tail, srcY)] * coefs[i];
            }

            dst[dstSt * dstY] = std::max(0, std::min(255, int(round(sum / sumCoefs[iTable]))));
        }
    }

}
