#include <cstring>
#include <vector>

#include "IQOLanczosResizerImpl.hpp"


namespace iqo {

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
    //! @param degree     Window size of Lanczos (ex. A=2 means Lanczos2)
    //! @param srcLen     Number of pixels of the source image
    //! @param dstLen     Number of pixels of the destination image
    //! @param dstOffset  The coordinate of the destination image
    //! @param pxScale  Scale of a pixel (ex. 2 when U plane of YUV420 image)
    //! @param tableLen   Size of table
    //! @param fTable     The table
    //! @return Sum of the table
    //!
    //! Calculate Lanczos coefficients from `-degree` to `+degree`.
    //!
    //! tableLen should be `2*degree` when up sampling.
    template<typename T>
    T setLanczosTable(
        int degree,
        size_t srcLen,
        size_t dstLen,
        ptrdiff_t dstOffset,
        size_t pxScale,
        int tableLen,
        T * __restrict fTable)
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

        for ( ptrdiff_t i = 0; i < tableLen; ++i ) {
            //     x = beginX + i * scale * pxScale
            double x = beginX + (i * dstLen * pxScale) / double(srcLen);
            T v = T(lanczos(degree, x));
            fTable[i] = v;
            fSum     += v;
        }

        return fSum;
    }

    template<>
    class LanczosResizerImpl<ArchGeneric> : public ILanczosResizerImpl
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
            size_t srcSt, const uint8_t * src,
            size_t dstSt, uint8_t * dst
        );

    private:
        void resizeYborder(
            ptrdiff_t srcSt, const uint8_t * src,
            ptrdiff_t dstW, float * dst,
            ptrdiff_t srcOY,
            const float * coefs,
            float * deno
        );
        void resizeYmain(
            ptrdiff_t srcSt, const uint8_t * src,
            ptrdiff_t dstW, float * dst,
            ptrdiff_t srcOY,
            const float * coefs
        );
        void resizeX(const float * src, uint8_t * dst);
        void resizeXborder(
            const float * src, uint8_t * dst,
            ptrdiff_t begin, ptrdiff_t end
        );
        void resizeXmain(
            const float * src, uint8_t * dst,
            ptrdiff_t begin, ptrdiff_t end
        );

        ptrdiff_t m_SrcW;
        ptrdiff_t m_SrcH;
        ptrdiff_t m_DstW;
        ptrdiff_t m_DstH;
        ptrdiff_t m_NumCoefsX;
        ptrdiff_t m_NumCoefsY;
        ptrdiff_t m_NumTablesX;
        ptrdiff_t m_NumTablesY;
        std::vector<float> m_TablesX; //!< Lanczos table * m_NumTablesX
        std::vector<float> m_TablesY; //!< Lanczos table * m_NumTablesY
        std::vector<float> m_Work;    //!< working memory
        std::vector<float> m_Nume;
        std::vector<float> m_Deno;
    };

    template<>
    bool LanczosResizerImpl_hasFeature<ArchGeneric>()
    {
        return true;
    }

    template<>
    ILanczosResizerImpl * LanczosResizerImpl_new<ArchGeneric>()
    {
        return new LanczosResizerImpl<ArchGeneric>();
    }

    // Constructor
    void LanczosResizerImpl<ArchGeneric>::init(
        unsigned int degree,
        size_t srcW, size_t srcH,
        size_t dstW, size_t dstH,
        size_t pxScale
    ) {
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
            // tableLen = 2*degree / scale
            size_t degree2 = std::max<size_t>(1, degree / pxScale);
            m_NumCoefsX = 2 * ptrdiff_t(std::ceil((degree2 * m_SrcW) / double(m_DstW)));
        }
        if ( m_SrcH <= m_DstH ) {
            // vertical: up-sampling
            m_NumCoefsY = 2 * degree;
        } else {
            // vertical: down-sampling
            // tableLen = 2*degree / scale
            size_t degree2 = std::max<size_t>(1, degree / pxScale);
            m_NumCoefsY = 2 * ptrdiff_t(std::ceil((degree2 * m_SrcH) / double(m_DstH)));
        }
        m_NumTablesX = m_DstW / gcd(m_SrcW, m_DstW);
        m_NumTablesY = m_DstH / gcd(m_SrcH, m_DstH);
        m_TablesX.reserve(m_NumCoefsX * m_NumTablesX);
        m_TablesX.resize(m_NumCoefsX * m_NumTablesX);
        m_TablesY.reserve(m_NumCoefsY * m_NumTablesY);
        m_TablesY.resize(m_NumCoefsY * m_NumTablesY);

        std::vector<float> tablesX(m_NumCoefsX);
        for ( ptrdiff_t dstX = 0; dstX < m_NumTablesX; ++dstX ) {
            float * table = &m_TablesX[dstX * m_NumCoefsX];
            double sumCoefs = setLanczosTable(degree, m_SrcW, m_DstW, dstX, pxScale, m_NumCoefsX, &tablesX[0]);
            for ( ptrdiff_t i = 0; i < m_NumCoefsX; ++i ) {
                table[i] = tablesX[i] / sumCoefs;
            }
        }
        std::vector<float> tablesY(m_NumCoefsY);
        for ( ptrdiff_t dstY = 0; dstY < m_NumTablesY; ++dstY ) {
            float * table = &m_TablesY[dstY * m_NumCoefsY];
            double sumCoefs = setLanczosTable(degree, m_SrcH, m_DstH, dstY, pxScale, m_NumCoefsY, &tablesY[0]);
            for ( ptrdiff_t i = 0; i < m_NumCoefsY; ++i ) {
                table[i] = tablesY[i] / sumCoefs;
            }
        }

        // allocate workspace
        m_Work.reserve(m_SrcW);
        m_Work.resize(m_SrcW);
        size_t maxW = std::max(m_SrcW, m_DstW);
        m_Nume.reserve(maxW);
        m_Nume.resize(maxW);
        m_Deno.reserve(maxW);
        m_Deno.resize(maxW);
    }

    void LanczosResizerImpl<ArchGeneric>::resize(
        size_t srcSt, const uint8_t * src,
        size_t dstSt, uint8_t * __restrict dst
    ) {
        // resize
        float * work = &m_Work[0];

        if ( m_SrcH == m_DstH ) {
            // resize only X axis
            for ( ptrdiff_t y = 0; y < m_SrcH; ++y ) {
                for ( ptrdiff_t x = 0; x < m_SrcW; ++x ) {
                    m_Work[m_SrcW * y + x] = src[srcSt * y + x];
                }
                resizeX(work, &dst[dstSt * y]);
            }

            return;
        }

        ptrdiff_t numCoefsOn2 = m_NumCoefsY / 2;
        //        mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstH / double(m_SrcH))
        ptrdiff_t mainBegin = ((numCoefsOn2 - 1) * m_DstH + m_SrcH-1) / m_SrcH;
        ptrdiff_t mainEnd = std::max<ptrdiff_t>(0, (m_SrcH - numCoefsOn2) * m_DstH / m_SrcH);
        const float * tablesY = &m_TablesY[0];
        ptrdiff_t tableSize = m_NumTablesY * m_NumCoefsY;
        ptrdiff_t iTable = 0;
        LinearIterator iSrcOY(m_DstH, m_SrcH);

        // border pixels
        for ( ptrdiff_t dstY = 0; dstY < mainBegin; ++dstY ) {
            //        srcOY = floor(dstY / scale) + 1
            ptrdiff_t srcOY = *iSrcOY++ + 1;
            //            coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
            const float * coefs = &tablesY[iTable];
            iTable += m_NumCoefsY;
            if ( iTable == tableSize ) {
                iTable = 0;
            }
            resizeYborder(
                srcSt, &src[0],
                m_SrcW, work,
                srcOY,
                coefs,
                &m_Deno[0]);
            resizeX(work, &dst[dstSt * dstY]);
        }

        // main loop
        for ( ptrdiff_t dstY = mainBegin; dstY < mainEnd; ++dstY ) {
            //        srcOY = floor(dstY / scale) + 1
            ptrdiff_t srcOY = *iSrcOY++ + 1;
            //            coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
            const float * coefs = &tablesY[iTable];
            iTable += m_NumCoefsY;
            if ( iTable == tableSize ) {
                iTable = 0;
            }
            resizeYmain(
                srcSt, &src[0],
                m_SrcW, work,
                srcOY,
                coefs);
            resizeX(work, &dst[dstSt * dstY]);
        }

        // border pixels
        for ( ptrdiff_t dstY = mainEnd; dstY < m_DstH; ++dstY ) {
            //        srcOY = floor(dstY / scale) + 1
            ptrdiff_t srcOY = *iSrcOY++ + 1;
            //            coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
            const float * coefs = &tablesY[iTable];
            iTable += m_NumCoefsY;
            if ( iTable == tableSize ) {
                iTable = 0;
            }
            resizeYborder(
                srcSt, &src[0],
                m_SrcW, work,
                srcOY,
                coefs,
                &m_Deno[0]);
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
    //! @param deno   Work memory for denominator
    void LanczosResizerImpl<ArchGeneric>::resizeYborder(
        ptrdiff_t srcSt, const uint8_t * src,
        ptrdiff_t dstW, float * __restrict dst,
        ptrdiff_t srcOY,
        const float * coefs,
        float * __restrict deno)
    {
        ptrdiff_t numCoefsOn2 = m_NumCoefsY / 2;
        float * nume = dst;

        std::memset(nume, 0, dstW * sizeof(*nume));
        std::memset(deno, 0, dstW * sizeof(*deno));

        for ( ptrdiff_t i = 0; i < m_NumCoefsY; ++i ) {
            float coef = coefs[i];
            for ( ptrdiff_t dstX = 0; dstX < dstW; ++dstX ) {
                ptrdiff_t srcY = srcOY - numCoefsOn2 + i;
                if ( 0 <= srcY && srcY < m_SrcH ) {
                    nume[dstX] += src[dstX + srcSt * srcY] * coef;
                    deno[dstX] += coef;
                }
            }
        }
        for ( ptrdiff_t dstX = 0; dstX < dstW; ++dstX ) {
            dst[dstX] = nume[dstX] / deno[dstX];
        }
    }

    //! resize vertical (main loop)
    //!
    //! @param srcSt  Stride in src (in byte)
    //! @param src    A row of source
    //! @param dst    A row of destination
    //! @param srcOY  The origin of current line
    //! @param coefs  The coefficients
    void LanczosResizerImpl<ArchGeneric>::resizeYmain(
        ptrdiff_t srcSt, const uint8_t * src,
        ptrdiff_t dstW, float * __restrict dst,
        ptrdiff_t srcOY,
        const float * coefs)
    {
        ptrdiff_t numCoefsOn2 = m_NumCoefsY / 2;

        std::memset(dst, 0, dstW * sizeof(*dst));

        for ( ptrdiff_t i = 0; i < m_NumCoefsY; ++i ) {
            float coef = coefs[i];
            for ( ptrdiff_t dstX = 0; dstX < dstW; ++dstX ) {
                ptrdiff_t srcY = srcOY - numCoefsOn2 + i;
                dst[dstX] += src[dstX + srcSt * srcY] * coef;
            }
        }
    }

    void LanczosResizerImpl<ArchGeneric>::resizeX(const float * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            // resize only Y axis
            ptrdiff_t dstW = m_DstW;
            for ( ptrdiff_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = clamp(0.0f, 255.0f, round(src[dstX]));
            }
            return;
        }

        ptrdiff_t numCoefsOn2 = m_NumCoefsX / 2;
        //       mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstW / double(m_SrcW))
        ptrdiff_t mainBegin = ((numCoefsOn2 - 1) * m_DstW + m_SrcW-1) / m_SrcW;
        ptrdiff_t mainEnd = std::max<ptrdiff_t>(0, (m_SrcW - numCoefsOn2) * m_DstW / m_SrcW);

        resizeXborder(src, dst, 0, mainBegin);
        resizeXmain(src, dst, mainBegin, mainEnd);
        resizeXborder(src, dst, mainEnd, m_DstW);
    }

    void LanczosResizerImpl<ArchGeneric>::resizeXborder(
        const float * src, uint8_t * __restrict dst,
        ptrdiff_t begin, ptrdiff_t end
    ) {
        ptrdiff_t numCoefsOn2 = m_NumCoefsX / 2;
        const float * tablesX = &m_TablesX[0];
        ptrdiff_t tableSize = m_NumTablesX * m_NumCoefsX;
        ptrdiff_t iTable = (m_NumCoefsX * begin) % tableSize;
        LinearIterator iSrcOX(m_DstW, m_SrcW);

        iSrcOX.setX(begin);
        for ( ptrdiff_t dstX = begin; dstX < end; ++dstX ) {
            //        srcOX = floor(dstX / scale) + 1;
            ptrdiff_t srcOX = *iSrcOX++ + 1;
            //            coefs = &tablesX[dstX % m_NumTablesX * m_NumCoefsX];
            const float * coefs = &tablesX[iTable];
            float nume = 0;
            float deno = 0;

            // iTable = (iTable + m_NumCoefsX) % tableSize;
            iTable += m_NumCoefsX;
            if ( iTable == tableSize ) {
                iTable = 0;
            }
            for ( ptrdiff_t i = 0; i < m_NumCoefsX; ++i ) {
                ptrdiff_t srcX = srcOX - numCoefsOn2 + i;
                if ( 0 <= srcX && srcX < m_SrcW ) {
                    float coef = coefs[i];
                    nume += src[srcX] * coef;
                    deno += coef;
                }
            }

            dst[dstX] = clamp(0.0f, 255.0f, nume / deno);
        }
    }

    //! resize horizontal (main loop)
    //!
    //! @param src    A row of source
    //! @param dst    A row of destination
    //! @param begin  Position of a first pixel
    //! @param end    Position of next of a last pixel
    void LanczosResizerImpl<ArchGeneric>::resizeXmain(
        const float * src, uint8_t * __restrict dst,
        ptrdiff_t begin, ptrdiff_t end)
    {
        ptrdiff_t numCoefsOn2 = m_NumCoefsX / 2;
        const float * tablesX = &m_TablesX[0];
        ptrdiff_t tableSize = m_NumTablesX * m_NumCoefsX;
        ptrdiff_t iTable = (m_NumCoefsX * begin) % tableSize;
        LinearIterator iSrcOX(m_DstW, m_SrcW);

        iSrcOX.setX(begin);
        for ( ptrdiff_t dstX = begin; dstX < end; ++dstX ) {
            //        srcOX = floor(dstX / scale) + 1;
            ptrdiff_t srcOX = *iSrcOX++ + 1;
            //            coefs = &tablesX[dstX % m_NumTablesX * m_NumCoefsX];
            const float * coefs = &tablesX[iTable];
            float sum = 0;

            // iTable = (iTable + m_NumCoefsX) % tableSize;
            iTable += m_NumCoefsX;
            if ( iTable == tableSize ) {
                iTable = 0;
            }
            for ( ptrdiff_t i = 0; i < m_NumCoefsX; ++i ) {
                ptrdiff_t srcX = srcOX - numCoefsOn2 + i;
                sum += src[srcX] * coefs[i];
            }

            dst[dstX] = clamp(0.0f, 255.0f, sum);
        }
    }

}
