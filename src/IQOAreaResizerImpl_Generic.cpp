#include <cstring>
#include <vector>

#include "IQOAreaResizerImpl.hpp"


namespace iqo {

    //! Calculate number of coefficients for area resampling
    size_t calcNumCoefsForArea(size_t srcLen, size_t dstLen)
    {
        // down-sampling
        //
        // case decimal number of coefs:
        //        scale = 5:4
        // num of coefs = ceil(5/4) = 2
        //
        //  |   o   |   o   |   o   |   o   |   o   |
        //  |    o    |    o    |    o    |    o    |
        //  +---------+
        // 0.0        +---------+
        //           1.25       +---------+
        //                     2.5        +---------+
        //                               3.75      5.0
        //
        // case decimal number of coefs:
        //        scale = 5:3
        // num of coefs = ceil(5/3) = 2 -> 3
        //  |  o  |  o  |  o  |  o  |  o  |
        //  |    o    |    o    |    o    |
        //  +---------+
        // 0.0        +---------+
        //           1.66  :    +---------+
        //                 :   3.33      5.0
        //                 :
        //              3 coefs
        //   (1.66-2.0, 2.0-3.0, 3.0-3.33)
        //
        // if ( lcm(1.0, scale-floor(scale)) > 1.0 ) { numCoefs += 1; }
        //
        // 3 times to source (3 is destination length)
        //  | o | o | o | o | o | o | o | o | o | o | o | o | o | o | o |
        //  |         o         |         o         |         o         |
        //  +-------------------+
        // 0.0          :       +-------------------+
        //             1.0     1.66                 +-------------------+
        //                                         3.33                5.0
        //
        // 1.66 * 3 = 5
        // floor(1.66) * 3 = (int(5) / int(3)) * 3 = 3
        // if ( lcm(5, 5 - (int(5) / int(3))*3) > 5 ) { numCoefs += 1; }

        size_t iScale = (srcLen / dstLen) * dstLen;             // floor(src/dst) * dst
        size_t numCoefs = alignCeil(srcLen, dstLen) / dstLen;   // ceil(src/dst)
        if ( lcm(srcLen, iScale) > ptrdiff_t(srcLen) ) {
            numCoefs++;
        }

        return numCoefs;
    }

    //! @brief Set Area table
    //! @param srcLen     Number of pixels of the source image
    //! @param dstLen     Number of pixels of the destination image
    //! @param dstOffset  The coordinate of the destination image
    //! @param numCoefs   Number of coefficients
    //! @param fTable     The table
    //! @return Sum of the table
    float setAreaTable(
        size_t srcLen,
        size_t dstLen,
        ptrdiff_t dstOffset,
        int numCoefs,
        float * __restrict fTable
    ) {

        double srcBeginX = ( dstOffset      * srcLen) / double(dstLen);
        double srcEndX   = ((dstOffset + 1) * srcLen) / double(dstLen);
        double srcX = srcBeginX;
        float fSum = 0;

        for ( ptrdiff_t i = 0; i < numCoefs; ++i ) {
            //     nextSrcX = std::min(srcEndX, std::floor(srcX + 1.0))
            double nextSrcX = std::min(srcEndX, std::floor(srcX) + 1.0);
            float v = float(nextSrcX - srcX);
            fTable[i] = v;
            fSum     += v;
            srcX = nextSrcX;
        }

        return fSum;
    }

    template<>
    class AreaResizerImpl<ArchGeneric> : public IAreaResizerImpl
    {
    public:
        //! Destructor
        virtual ~AreaResizerImpl() {}

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
        //! round(a / b)
        static int roundedDiv(int a, int b, int biasbit)
        {
            const int k0_5 = (1 << biasbit) / 2;
            return (a + k0_5) / b;
        }

        //! fixed_point_to_int(round(a))
        static int convertToInt(int a, int biasbit)
        {
            const int k0_5 = (1 << biasbit) / 2;
            return (a + k0_5) >> biasbit;
        }

        //! dst[i] = src[i] * kBias / srcSum (src will be broken)
        void adjustCoefs(
            float * srcBegin, float * srcEnd,
            float srcSum,
            int bias,
            int16_t * dst
        );

        void resizeYmain(
            ptrdiff_t srcSt, const uint8_t * src,
            ptrdiff_t dstW, int16_t * dst,
            ptrdiff_t srcOY,
            const int16_t * coefs
        );
        void resizeX(const int16_t * src, uint8_t * dst);
        void resizeXmain(const int16_t * src, uint8_t * dst);

        enum {
            // for fixed point
            kBiasBit = 6,
            kBias    = 1 << kBiasBit,

            kBias15Bit = 15,
            kBias15  = 1 << kBias15Bit,
        };

        ptrdiff_t m_SrcW;
        ptrdiff_t m_SrcH;
        ptrdiff_t m_DstW;
        ptrdiff_t m_DstH;
        ptrdiff_t m_NumCoefsX;
        ptrdiff_t m_NumCoefsY;
        ptrdiff_t m_NumTablesX;
        ptrdiff_t m_NumTablesY;
        std::vector<int16_t> m_TablesX; //!< Area table * m_NumTablesX
        std::vector<int16_t> m_TablesY; //!< Area table * m_NumTablesY
        std::vector<int16_t> m_Work;    //!< working memory
        std::vector<int16_t> m_Nume;
        std::vector<int16_t> m_Deno;
    };

    template<>
    bool AreaResizerImpl_hasFeature<ArchGeneric>()
    {
        return true;
    }

    template<>
    IAreaResizerImpl * AreaResizerImpl_new<ArchGeneric>()
    {
        return new AreaResizerImpl<ArchGeneric>();
    }

    // Constructor
    void AreaResizerImpl<ArchGeneric>::init(
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
        m_NumCoefsX = calcNumCoefsForArea(rSrcW, rDstW);
        m_NumCoefsY = calcNumCoefsForArea(rSrcH, rDstH);
        m_NumTablesX = rDstW;
        m_NumTablesY = rDstH;
        m_TablesX.reserve(m_NumCoefsX * m_NumTablesX);
        m_TablesX.resize(m_NumCoefsX * m_NumTablesX);
        m_TablesY.reserve(m_NumCoefsY * m_NumTablesY);
        m_TablesY.resize(m_NumCoefsY * m_NumTablesY);

        std::vector<float> tablesX(m_NumCoefsX);
        for ( ptrdiff_t dstX = 0; dstX < m_NumTablesX; ++dstX ) {
            int16_t * table = &m_TablesX[dstX * m_NumCoefsX];
            double sumCoefs = setAreaTable(rSrcW, rDstW, dstX, m_NumCoefsX, &tablesX[0]);
            adjustCoefs(&tablesX[0], &tablesX[m_NumCoefsX], sumCoefs, kBias15, &table[0]);
        }
        std::vector<float> tablesY(m_NumCoefsY);
        for ( ptrdiff_t dstY = 0; dstY < m_NumTablesY; ++dstY ) {
            int16_t * table = &m_TablesY[dstY * m_NumCoefsY];
            double sumCoefs = setAreaTable(rSrcH, rDstH, dstY, m_NumCoefsY, &tablesY[0]);
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

    void AreaResizerImpl<ArchGeneric>::adjustCoefs(
        float * __restrict srcBegin, float * __restrict srcEnd,
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

    void AreaResizerImpl<ArchGeneric>::resize(
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

        ptrdiff_t numCoefs = m_NumCoefsY;
        const int16_t * tablesY = &m_TablesY[0];
        ptrdiff_t tableSize = m_NumTablesY * m_NumCoefsY;
        ptrdiff_t iTable = 0;
        LinearIterator iSrcOY(m_DstH, m_SrcH);
        ptrdiff_t dstH = m_DstH;

        // main loop
        for ( ptrdiff_t dstY = 0; dstY < dstH; ++dstY ) {
            //        srcOY = floor(dstY / scale);
            ptrdiff_t srcOY = *iSrcOY++;
            //              coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
            const int16_t * coefs = &tablesY[iTable];
            iTable += numCoefs;
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
    }

    //! resize vertical (main loop)
    //!
    //! @param srcSt  Stride in src (in byte)
    //! @param src    A row of source
    //! @param dst    A row of destination (multiplied by kBias)
    //! @param srcOY  The origin of current line
    //! @param coefs  The coefficients (multiplied by kBias)
    void AreaResizerImpl<ArchGeneric>::resizeYmain(
        ptrdiff_t srcSt, const uint8_t * src,
        ptrdiff_t dstW, int16_t * __restrict dst,
        ptrdiff_t srcOY,
        const int16_t * coefs)
    {
        ptrdiff_t numCoefs = m_NumCoefsY;

        std::memset(dst, 0, dstW * sizeof(*dst));

        for ( ptrdiff_t i = 0; i < numCoefs; ++i ) {
            int16_t coef = coefs[i];
            for ( ptrdiff_t dstX = 0; dstX < dstW; ++dstX ) {
                ptrdiff_t srcY = srcOY + i;
                dst[dstX] += src[dstX + srcSt * srcY] * coef;
            }
        }
    }

    void AreaResizerImpl<ArchGeneric>::resizeX(const int16_t * src, uint8_t * __restrict dst)
    {
        if ( m_SrcW == m_DstW ) {
            // resize only Y axis
            ptrdiff_t dstW = m_DstW;
            for ( ptrdiff_t dstX = 0; dstX < dstW; dstX++ ) {
                dst[dstX] = convertToInt(src[dstX], kBiasBit);
            }
            return;
        }

        resizeXmain(src, dst);
    }

    //! resize horizontal (main loop)
    //!
    //! @param src    A row of source (multiplied by kBias)
    //! @param dst    A row of destination
    void AreaResizerImpl<ArchGeneric>::resizeXmain(const int16_t * src, uint8_t * __restrict dst)
    {
        ptrdiff_t numCoefs = m_NumCoefsX;
        const int16_t * tablesX = &m_TablesX[0];
        ptrdiff_t tableSize = m_NumTablesX * m_NumCoefsX;
        ptrdiff_t iTable = 0;
        LinearIterator iSrcOX(m_DstW, m_SrcW);
        ptrdiff_t dstW = m_DstW;

        for ( ptrdiff_t dstX = 0; dstX < dstW; ++dstX ) {
            //        srcOX = floor(dstX / scale);
            ptrdiff_t srcOX = *iSrcOX++;
            //              coefs = &tablesX[dstX % m_NumTablesX * m_NumCoefsX];
            const int16_t * coefs = &tablesX[iTable];
            int sum = 0;

            // iTable = (iTable + m_NumCoefsX) % tableSize;
            iTable += numCoefs;
            if ( iTable == tableSize ) {
                iTable = 0;
            }
            for ( ptrdiff_t i = 0; i < numCoefs; ++i ) {
                ptrdiff_t srcX = srcOX + i;
                sum += src[srcX] * coefs[i];
            }

            dst[dstX] = clamp<int16_t>(0, 255, convertToInt(sum, kBias15Bit+kBiasBit));
        }
    }

}
