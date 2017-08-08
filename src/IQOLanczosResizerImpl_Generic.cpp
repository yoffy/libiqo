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

    //! Calculate number of coefficients for Lanczos resampling
    size_t calcNumCoefsForLanczos(int degree, size_t srcLen, size_t dstLen, size_t pxScale)
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

        size_t numCoefs;

        if ( srcLen <= dstLen ) {
            // horizontal: up-sampling
            numCoefs = 2 * degree;
        } else {
            // vertical: down-sampling
            // tableLen = 2*degree / scale
            size_t degree2 = std::max<size_t>(1, degree / pxScale);
            numCoefs = 2 * ptrdiff_t(std::ceil((degree2 * srcLen) / double(dstLen)));
        }

        return numCoefs;
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
    float setLanczosTable(
        int degree,
        size_t srcLen,
        size_t dstLen,
        ptrdiff_t dstOffset,
        size_t pxScale,
        ptrdiff_t numCoefs,
        float * __restrict fTable)
    {
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
            int degFactor = std::max<int>(1, int(pxScale) / degree);
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

        float fSum = 0;

        for ( ptrdiff_t i = 0; i < numCoefs; ++i ) {
            //     x = beginX + i * scale * pxScale
            double x = beginX + (i * dstLen * pxScale) / double(srcLen);
            float v = float(lanczos(degree, x));
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
        //! round(a / b)
        static int16_t roundedDiv(int a, int b, int biasbit)
        {
            const int k0_5 = (1 << biasbit) / 2;
            return int16_t((a + k0_5) / b);
        }

        //! fixed_point_to_int(round(a))
        static int16_t convertToInt(int a, int biasbit)
        {
            const int k0_5 = (1 << biasbit) / 2;
            return int16_t((a + k0_5) >> biasbit);
        }

        //! dst[i] = src[i] * kBias / srcSum (src will be broken)
        void adjustCoefs(
            float * srcBegin, float * srcEnd,
            float srcSum,
            int bias,
            int16_t * dst
        );

        void resizeYborder(
            ptrdiff_t srcSt, const uint8_t * src,
            ptrdiff_t dstW, int16_t * dst,
            ptrdiff_t srcOY,
            const int16_t * coefs,
            int16_t * deno
        );
        void resizeYmain(
            ptrdiff_t srcSt, const uint8_t * src,
            ptrdiff_t dstW, int16_t * dst,
            ptrdiff_t srcOY,
            const int16_t * coefs
        );
        void resizeX(const int16_t * src, uint8_t * dst);
        void resizeXborder(
            const int16_t * src, uint8_t * dst,
            ptrdiff_t begin, ptrdiff_t end
        );
        void resizeXmain(
            const int16_t * src, uint8_t * dst,
            ptrdiff_t begin, ptrdiff_t end
        );

        enum {
            // for fixed point
            kBiasBit = 6,
            kBias    = 1 << kBiasBit,

            kBias14Bit = 14,
            kBias14  = 1 << kBias14Bit,
        };

        ptrdiff_t m_SrcW;
        ptrdiff_t m_SrcH;
        ptrdiff_t m_DstW;
        ptrdiff_t m_DstH;
        ptrdiff_t m_NumCoefsX;
        ptrdiff_t m_NumCoefsY;
        ptrdiff_t m_NumTablesX;
        ptrdiff_t m_NumTablesY;
        std::vector<int16_t> m_TablesX; //!< Lanczos table * m_NumTablesX
        std::vector<int16_t> m_TablesY; //!< Lanczos table * m_NumTablesY
        std::vector<int16_t> m_Work;    //!< working memory
        std::vector<int16_t> m_Nume;
        std::vector<int16_t> m_Deno;
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
        size_t gcdW = gcd(m_SrcW, m_DstW);
        size_t gcdH = gcd(m_SrcH, m_DstH);
        size_t rSrcW = m_SrcW / gcdW;
        size_t rDstW = m_DstW / gcdW;
        size_t rSrcH = m_SrcH / gcdH;
        size_t rDstH = m_DstH / gcdH;
        m_NumCoefsX = calcNumCoefsForLanczos(degree, rSrcW, rDstW, pxScale);
        m_NumCoefsY = calcNumCoefsForLanczos(degree, rSrcH, rDstH, pxScale);
        m_NumTablesX = rDstW;
        m_NumTablesY = rDstH;
        m_TablesX.reserve(m_NumCoefsX * m_NumTablesX);
        m_TablesX.resize(m_NumCoefsX * m_NumTablesX);
        m_TablesY.reserve(m_NumCoefsY * m_NumTablesY);
        m_TablesY.resize(m_NumCoefsY * m_NumTablesY);

        std::vector<float> tablesX(m_NumCoefsX);
        for ( ptrdiff_t dstX = 0; dstX < m_NumTablesX; ++dstX ) {
            int16_t * table = &m_TablesX[dstX * m_NumCoefsX];
            float sumCoefs = setLanczosTable(degree, rSrcW, rDstW, dstX, pxScale, m_NumCoefsX, &tablesX[0]);
            adjustCoefs(&tablesX[0], &tablesX[m_NumCoefsX], sumCoefs, kBias14, &table[0]);
        }
        std::vector<float> tablesY(m_NumCoefsY);
        for ( ptrdiff_t dstY = 0; dstY < m_NumTablesY; ++dstY ) {
            int16_t * table = &m_TablesY[dstY * m_NumCoefsY];
            float sumCoefs = setLanczosTable(degree, rSrcH, rDstH, dstY, pxScale, m_NumCoefsY, &tablesY[0]);
            adjustCoefs(&tablesY[0], &tablesY[m_NumCoefsY], sumCoefs, kBias, &table[0]);
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

    void LanczosResizerImpl<ArchGeneric>::adjustCoefs(
        float * __restrict srcBegin, float * __restrict srcEnd,
        float srcSum,
        int bias,
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

    void LanczosResizerImpl<ArchGeneric>::resize(
        size_t srcSt, const uint8_t * src,
        size_t dstSt, uint8_t * __restrict dst
    ) {
        // resize
        int16_t * work = &m_Work[0];

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
        const int16_t * tablesY = &m_TablesY[0];
        ptrdiff_t tableSize = m_NumTablesY * m_NumCoefsY;
        ptrdiff_t iTable = 0;
        LinearIterator iSrcOY(m_DstH, m_SrcH);

        // border pixels
        for ( ptrdiff_t dstY = 0; dstY < mainBegin; ++dstY ) {
            //        srcOY = floor(dstY / scale) + 1
            ptrdiff_t srcOY = *iSrcOY++ + 1;
            //              coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
            const int16_t * coefs = &tablesY[iTable];
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
            //              coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
            const int16_t * coefs = &tablesY[iTable];
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
            //              coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
            const int16_t * coefs = &tablesY[iTable];
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
    //! @param dst    A row of destination (multiplied by kBias)
    //! @param srcOY  The origin of current line
    //! @param coefs  The coefficients (multiplied by kBias)
    //! @param deno   Work memory for denominator
    void LanczosResizerImpl<ArchGeneric>::resizeYborder(
        ptrdiff_t srcSt, const uint8_t * src,
        ptrdiff_t dstW, int16_t * __restrict dst,
        ptrdiff_t srcOY,
        const int16_t * coefs,
        int16_t * __restrict deno)
    {
        ptrdiff_t numCoefsOn2 = m_NumCoefsY / 2;
        int16_t * nume = dst;

        std::memset(nume, 0, dstW * sizeof(*nume));
        std::memset(deno, 0, dstW * sizeof(*deno));

        for ( ptrdiff_t i = 0; i < m_NumCoefsY; ++i ) {
            int16_t coef = coefs[i];
            for ( ptrdiff_t dstX = 0; dstX < dstW; ++dstX ) {
                ptrdiff_t srcY = srcOY - numCoefsOn2 + i;
                if ( 0 <= srcY && srcY < m_SrcH ) {
                    nume[dstX] = int16_t(nume[dstX] + src[dstX + srcSt * srcY] * coef);
                    deno[dstX] = int16_t(deno[dstX] + coef);
                }
            }
        }
        for ( ptrdiff_t dstX = 0; dstX < dstW; ++dstX ) {
            dst[dstX] = int16_t(int(nume[dstX])*kBias / deno[dstX]);
        }
    }

    //! resize vertical (main loop)
    //!
    //! @param srcSt  Stride in src (in byte)
    //! @param src    A row of source
    //! @param dst    A row of destination (multiplied by kBias)
    //! @param srcOY  The origin of current line
    //! @param coefs  The coefficients (multiplied by kBias)
    void LanczosResizerImpl<ArchGeneric>::resizeYmain(
        ptrdiff_t srcSt, const uint8_t * src,
        ptrdiff_t dstW, int16_t * __restrict dst,
        ptrdiff_t srcOY,
        const int16_t * coefs)
    {
        ptrdiff_t numCoefsOn2 = m_NumCoefsY / 2;

        std::memset(dst, 0, dstW * sizeof(*dst));

        for ( ptrdiff_t i = 0; i < m_NumCoefsY; ++i ) {
            int16_t coef = coefs[i];
            for ( ptrdiff_t dstX = 0; dstX < dstW; ++dstX ) {
                ptrdiff_t srcY = srcOY - numCoefsOn2 + i;
                dst[dstX] = int16_t(dst[dstX] + src[dstX + srcSt * srcY] * coef);
            }
        }
    }

    void LanczosResizerImpl<ArchGeneric>::resizeX(const int16_t * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            // resize only Y axis
            ptrdiff_t dstW = m_DstW;
            for ( ptrdiff_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = uint8_t(clamp<int16_t>(0, 255, convertToInt(src[dstX], kBiasBit)));
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
        const int16_t * src, uint8_t * __restrict dst,
        ptrdiff_t begin, ptrdiff_t end
    ) {
        ptrdiff_t numCoefsOn2 = m_NumCoefsX / 2;
        const int16_t * tablesX = &m_TablesX[0];
        ptrdiff_t tableSize = m_NumTablesX * m_NumCoefsX;
        ptrdiff_t iTable = (m_NumCoefsX * begin) % tableSize;
        LinearIterator iSrcOX(m_DstW, m_SrcW);

        iSrcOX.setX(begin);
        for ( ptrdiff_t dstX = begin; dstX < end; ++dstX ) {
            //        srcOX = floor(dstX / scale) + 1;
            ptrdiff_t srcOX = *iSrcOX++ + 1;
            //              coefs = &tablesX[dstX % m_NumTablesX * m_NumCoefsX];
            const int16_t * coefs = &tablesX[iTable];
            int nume = 0;
            int deno = 0;

            // iTable = (iTable + m_NumCoefsX) % tableSize;
            iTable += m_NumCoefsX;
            if ( iTable == tableSize ) {
                iTable = 0;
            }
            for ( ptrdiff_t i = 0; i < m_NumCoefsX; ++i ) {
                ptrdiff_t srcX = srcOX - numCoefsOn2 + i;
                if ( 0 <= srcX && srcX < m_SrcW ) {
                    int16_t coef = coefs[i];
                    nume += src[srcX] * coef;
                    deno += coef;
                }
            }

            dst[dstX] = uint8_t(clamp<int16_t>(0, 255, roundedDiv(nume, deno*kBias, kBias14Bit+kBiasBit)));
        }
    }

    //! resize horizontal (main loop)
    //!
    //! @param src    A row of source (multiplied by kBias)
    //! @param dst    A row of destination
    //! @param begin  Position of a first pixel
    //! @param end    Position of next of a last pixel
    void LanczosResizerImpl<ArchGeneric>::resizeXmain(
        const int16_t * src, uint8_t * __restrict dst,
        ptrdiff_t begin, ptrdiff_t end)
    {
        ptrdiff_t numCoefsOn2 = m_NumCoefsX / 2;
        const int16_t * tablesX = &m_TablesX[0];
        ptrdiff_t tableSize = m_NumTablesX * m_NumCoefsX;
        ptrdiff_t iTable = (m_NumCoefsX * begin) % tableSize;
        LinearIterator iSrcOX(m_DstW, m_SrcW);

        iSrcOX.setX(begin);
        for ( ptrdiff_t dstX = begin; dstX < end; ++dstX ) {
            //        srcOX = floor(dstX / scale) + 1;
            ptrdiff_t srcOX = *iSrcOX++ + 1;
            //              coefs = &tablesX[dstX % m_NumTablesX * m_NumCoefsX];
            const int16_t * coefs = &tablesX[iTable];
            int sum = 0;

            // iTable = (iTable + m_NumCoefsX) % tableSize;
            iTable += m_NumCoefsX;
            if ( iTable == tableSize ) {
                iTable = 0;
            }
            for ( ptrdiff_t i = 0; i < m_NumCoefsX; ++i ) {
                ptrdiff_t srcX = srcOX - numCoefsOn2 + i;
                sum += src[srcX] * coefs[i];
            }

            dst[dstX] = uint8_t(clamp<int16_t>(0, 255, convertToInt(sum, kBias14Bit+kBiasBit)));
        }
    }

}
