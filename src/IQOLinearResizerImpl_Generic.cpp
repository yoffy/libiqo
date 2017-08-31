#include <cstring>
#include <vector>

#include "IQOLinearResizerImpl.hpp"
#include "math.hpp"


namespace iqo {

    //! @brief Set Linear table
    //! @param srcLen     Number of pixels of the source image
    //! @param dstLen     Number of pixels of the destination image
    //! @param fTable     The table
    //! @return Sum of the table
    void setLinearTable(
        size_t srcLen,
        size_t dstLen,
        float * __restrict fTable
    ) {
        // o: center of a pixel (on a coordinate)
        // |: boundary of pixels
        //
        // scale = 4:5
        //
        // fomula dstX to srcX (align center):
        //     (4 - 5)/(2 * 5) + dstX * 4/5
        //     = (dstX+0.5) * 4/5 - 0.5
        //
        // srcX -0.5  0.00 0.50 1.00 1.5  2.00 2.50 3.00 3.5
        // src    |    o    |    o    |    o    |    o    |
        // dst    |   o   |   o   |   o   |   o   |   o   |
        // dstX -0.5 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5
        // srcX     -0.1     0.7     1.5     2.3     3.1
        //
        //
        // scale = 2:11
        // fomula dstX to srcX (align center):
        //     (2 - 11)/(2 * 11) + dstX * 2/11
        //     = (dstX+0.5) * 2/11 - 0.5
        //
        // srcX -0.5        0.0        0.5        1.0        1.5
        // src    |          o          |          o          |
        // dst    | o | o | o | o | o | o | o | o | o | o | o |
        // dstX     0   1   2   3   4   5   6   7   8   9  10
        // srcX   -0.4-0.2-0.0 0.1 0.3 0.5 0.7 0.9 1.0 1.2 1.4
        //
        for ( size_t i = 0; i < dstLen; i++ ) {
            double x = double(i);
            //                                              +0.5 = -0.5 + 1.0 for modf(neg to pos)
            float coef1 = float(std::modf((x+0.5) * srcLen/dstLen + 0.5, &x));
            float coef0 = 1.0f - coef1;
            fTable[i * 2 + 0] = coef0;
            fTable[i * 2 + 1] = coef1;
        }
    }

    template<>
    class LinearResizerImpl<ArchGeneric> : public ILinearResizerImpl
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
            size_t srcSt, const uint8_t * src,
            size_t dstSt, uint8_t * dst
        );

    private:
        //! fixed_point_to_int(round(a))
        static int16_t convertToInt(int a, int biasbit)
        {
            const int k0_5 = (1 << biasbit) / 2;
            return int16_t((a + k0_5) >> biasbit);
        }

        //! dst[i] = src[i] * kBias / srcSum (src will be broken)
        void adjustCoefs(
            float * srcBegin, float * srcEnd,
            uint16_t bias,
            uint16_t * dst
        );

        void resizeYborder(
            ptrdiff_t srcSt, const uint8_t * src,
            ptrdiff_t dstW, uint16_t * dst,
            ptrdiff_t srcOY
        );
        void resizeYmain(
            ptrdiff_t srcSt, const uint8_t * src,
            ptrdiff_t dstW, uint16_t * dst,
            ptrdiff_t srcOY,
            const uint16_t * coefs
        );
        void resizeX(const uint16_t * src, uint8_t * dst);
        void resizeXborder(
            const uint16_t * src, uint8_t * dst,
            ptrdiff_t srcX,
            ptrdiff_t begin, ptrdiff_t end
        );
        void resizeXmain(
            const uint16_t * src, uint8_t * dst,
            ptrdiff_t begin, ptrdiff_t end
        );

        enum {
            m_NumCoefsX = 2,
            m_NumCoefsY = 2,

            // for fixed point
            kBiasBit = 8,
            kBias    = 1 << kBiasBit,

            kBias15Bit = 15,
            kBias15    = 1 << kBias15Bit,
        };

        ptrdiff_t m_SrcW;
        ptrdiff_t m_SrcH;
        ptrdiff_t m_DstW;
        ptrdiff_t m_DstH;
        ptrdiff_t m_NumTablesX;
        ptrdiff_t m_NumTablesY;
        std::vector<uint16_t> m_TablesX;    //!< 2 * m_NumTablesX
        std::vector<uint16_t> m_TablesY;    //!< 2 * m_NumTablesY
        std::vector<uint16_t> m_Work;       //!< working memory
    };

    template<>
    bool LinearResizerImpl_hasFeature<ArchGeneric>()
    {
        return true;
    }

    template<>
    ILinearResizerImpl * LinearResizerImpl_new<ArchGeneric>()
    {
        return new LinearResizerImpl<ArchGeneric>();
    }

    // Constructor
    void LinearResizerImpl<ArchGeneric>::init(
        size_t srcW, size_t srcH,
        size_t dstW, size_t dstH
    ) {
        m_SrcW = srcW;
        m_SrcH = srcH;
        m_DstW = dstW;
        m_DstH = dstH;

        // setup coefficients
        size_t gcdW = gcd(m_SrcW, m_DstW);
        size_t gcdH = gcd(m_SrcH, m_DstH);
        size_t rSrcW = m_SrcW / gcdW;   // reduction
        size_t rDstW = m_DstW / gcdW;
        size_t rSrcH = m_SrcH / gcdH;
        size_t rDstH = m_DstH / gcdH;
        m_NumTablesX = rDstW;
        m_NumTablesY = rDstH;
        m_TablesX.reserve(m_NumCoefsX * m_NumTablesX);
        m_TablesX.resize(m_NumCoefsX * m_NumTablesX);
        m_TablesY.reserve(m_NumCoefsY * m_NumTablesY);
        m_TablesY.resize(m_NumCoefsY * m_NumTablesY);

        std::vector<float> tablesX(m_NumCoefsX * m_NumTablesX);
        setLinearTable(rSrcW, rDstW, &tablesX[0]);
        adjustCoefs(&tablesX[0], &tablesX[m_NumCoefsX * m_NumTablesX], kBias15, &m_TablesX[0]);

        std::vector<float> tablesY(m_NumCoefsY * m_NumTablesY);
        setLinearTable(rSrcH, rDstH, &tablesY[0]);
        adjustCoefs(&tablesY[0], &tablesY[m_NumCoefsY * m_NumTablesY], kBias, &m_TablesY[0]);

        // allocate workspace
        m_Work.reserve(m_SrcW);
        m_Work.resize(m_SrcW);
    }

    void LinearResizerImpl<ArchGeneric>::adjustCoefs(
        float * __restrict srcBegin, float * __restrict srcEnd,
        uint16_t bias,
        uint16_t * __restrict dst)
    {
        const uint16_t k1_0 = bias;
        size_t numCoefs = 2;
        size_t numTables = (srcEnd - srcBegin) / numCoefs;

        for ( size_t i = 0; i < numTables; ++i ) {
            uint16_t coef0 = uint16_t(round(srcBegin[2*i] * bias));
            uint16_t coef1 = k1_0 - coef0;
            dst[2*i + 0] = coef0;
            dst[2*i + 1] = coef1;
        }
    }

    void LinearResizerImpl<ArchGeneric>::resize(
        size_t srcSt, const uint8_t * src,
        size_t dstSt, uint8_t * __restrict dst
    ) {
        ptrdiff_t  srcW = m_SrcW;
        ptrdiff_t  srcH = m_SrcH;
        ptrdiff_t  dstH = m_DstH;
        uint16_t * work = &m_Work[0];

        if ( m_SrcH == m_DstH ) {
            // resize only X axis
            for ( ptrdiff_t y = 0; y < srcH; ++y ) {
                for ( ptrdiff_t x = 0; x < srcW; ++x ) {
                    work[x] = uint16_t(src[srcSt * y + x] * kBias);
                }
                resizeX(work, &dst[dstSt * y]);
            }

            return;
        }

        ptrdiff_t        numCoefs = 2; // = m_NumCoefsY;
        const uint16_t * tablesY = &m_TablesY[0];
        ptrdiff_t        tableSize = m_NumTablesY * numCoefs;
        ptrdiff_t        iTable = 0;
        LinearIterator   iSrcOY(dstH, srcH);
        double           fMainBegin = std::ceil(0.5 * dstH / srcH - 0.5);
        ptrdiff_t        mainBegin  = clamp<ptrdiff_t>(0, dstH, ptrdiff_t(fMainBegin));
        ptrdiff_t        mainEnd    = clamp<ptrdiff_t>(0, dstH, dstH - mainBegin);

        // border pixels
        for ( ptrdiff_t dstY = 0; dstY < mainBegin; ++dstY ) {
            ptrdiff_t srcOY = 0;
            resizeYborder(
                srcSt, &src[0],
                srcW, work,
                srcOY);
            resizeX(work, &dst[dstSt * dstY]);
        }

        // main loop

        // align center
        iSrcOY.setX(srcH - dstH, 2 * dstH);
        iSrcOY += mainBegin;
        iTable = mainBegin % m_NumTablesY * numCoefs;
        for ( ptrdiff_t dstY = mainBegin; dstY < mainEnd; ++dstY ) {
            //        srcOY = floor((dstY+0.5) / scale - 0.5);
            ptrdiff_t srcOY = *iSrcOY++;
            //               coefs = &tablesY[dstY % m_NumTablesY * numCoefs];
            const uint16_t * coefs = &tablesY[iTable];
            iTable += numCoefs;
            if ( iTable == tableSize ) {
                iTable = 0;
            }
            resizeYmain(
                srcSt, &src[0],
                srcW, work,
                srcOY,
                coefs);
            resizeX(work, &dst[dstSt * dstY]);
        }

        // border pixels
        for ( ptrdiff_t dstY = mainEnd; dstY < dstH; ++dstY ) {
            ptrdiff_t srcOY = srcH - 1;
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
    void LinearResizerImpl<ArchGeneric>::resizeYborder(
        ptrdiff_t srcSt, const uint8_t * src,
        ptrdiff_t dstW, uint16_t * __restrict dst,
        ptrdiff_t srcOY
    ) {
        uint16_t coef = kBias;
        for ( ptrdiff_t dstX = 0; dstX < dstW; ++dstX ) {
            dst[dstX] = uint16_t(src[dstX + srcSt * srcOY] * coef);
        }
    }

    //! resize vertical (main loop)
    //!
    //! @param srcSt  Stride in src (in byte)
    //! @param src    A row of source
    //! @param dst    A row of destination (multiplied by kBias)
    //! @param srcOY  The origin of current line
    //! @param coefs  The coefficients (multiplied by kBias)
    void LinearResizerImpl<ArchGeneric>::resizeYmain(
        ptrdiff_t srcSt, const uint8_t * src,
        ptrdiff_t dstW, uint16_t * __restrict dst,
        ptrdiff_t srcOY,
        const uint16_t * coefs)
    {
        ptrdiff_t numCoefs = 2; // = m_NumCoefsY;

        std::memset(dst, 0, dstW * sizeof(*dst));

        for ( ptrdiff_t i = 0; i < numCoefs; ++i ) {
            uint16_t coef = coefs[i];
            ptrdiff_t srcY = srcOY + i;
            for ( ptrdiff_t dstX = 0; dstX < dstW; ++dstX ) {
                dst[dstX] = uint16_t(dst[dstX] + src[dstX + srcSt * srcY] * coef);
            }
        }
    }

    void LinearResizerImpl<ArchGeneric>::resizeX(const uint16_t * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            // resize only Y axis
            ptrdiff_t dstW = m_DstW;
            for ( ptrdiff_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = uint8_t(clamp<int16_t>(0, 255, convertToInt(src[dstX], kBiasBit)));
            }
            return;
        }

        ptrdiff_t   dstW        = m_DstW;
        double      fMainBegin  = std::ceil(0.5 * dstW / m_SrcW - 0.5);
        ptrdiff_t   mainBegin   = clamp<ptrdiff_t>(0, dstW, ptrdiff_t(fMainBegin));
        ptrdiff_t   mainEnd     = clamp<ptrdiff_t>(0, dstW, dstW - mainBegin);

        resizeXborder(src, dst, 0, 0, mainBegin);
        resizeXmain(src, dst, mainBegin, mainEnd);
        resizeXborder(src, dst, m_SrcW - 1, mainEnd, dstW);
    }

    //! resize horizontal (border loop)
    //!
    //! @param src    A row of source (multiplied by kBias)
    //! @param dst    A row of destination
    //! @param srcX   Position of a source pixel (a first or a last pixel)
    //! @param begin  Position of a first pixel
    //! @param end    Position of next of a last pixel
    void LinearResizerImpl<ArchGeneric>::resizeXborder(
        const uint16_t * src, uint8_t * __restrict dst,
        ptrdiff_t srcX,
        ptrdiff_t begin, ptrdiff_t end
    ) {
        uint16_t vSrc = src[srcX];
        uint8_t  vDst = uint8_t(clamp<uint16_t>(0, 255, convertToInt(vSrc, kBiasBit)));

        for ( ptrdiff_t dstX = begin; dstX < end; ++dstX ) {
            dst[dstX] = vDst;
        }
    }

    //! resize horizontal (main loop)
    //!
    //! @param src    A row of source (multiplied by kBias)
    //! @param dst    A row of destination
    //! @param begin  Position of a first pixel
    //! @param end    Position of next of a last pixel
    void LinearResizerImpl<ArchGeneric>::resizeXmain(
        const uint16_t * src, uint8_t * __restrict dst,
        ptrdiff_t begin, ptrdiff_t end
    ) {
        ptrdiff_t numCoefs = 2; // = m_NumCoefsX;
        const uint16_t * tablesX = &m_TablesX[0];
        ptrdiff_t tableSize = m_NumTablesX * numCoefs;
        ptrdiff_t iTable = (numCoefs * begin) % tableSize;
        LinearIterator iSrcOX(m_DstW, m_SrcW);

        // align center
        iSrcOX.setX(m_SrcW - m_DstW, 2 * m_DstW);
        iSrcOX += begin;

        for ( ptrdiff_t dstX = begin; dstX < end; ++dstX ) {
            //        srcOX = floor((dstX+0.5) / scale - 0.5);
            ptrdiff_t srcOX = *iSrcOX++;
            //               coefs = &tablesX[dstX % m_NumTablesX * m_NumCoefsX];
            const uint16_t * coefs = &tablesX[iTable];
            int32_t sum = 0;

            // iTable = (iTable + m_NumCoefsX) % tableSize;
            iTable += numCoefs;
            if ( iTable == tableSize ) {
                iTable = 0;
            }
            for ( ptrdiff_t i = 0; i < numCoefs; ++i ) {
                ptrdiff_t srcX = srcOX + i;
                sum += src[srcX] * int32_t(coefs[i]);
            }

            dst[dstX] = uint8_t(clamp<uint16_t>(0, 255, convertToInt(sum, kBias15Bit+kBiasBit)));
        }
    }

}
