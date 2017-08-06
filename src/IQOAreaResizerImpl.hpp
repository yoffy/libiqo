#include <stdint.h>
#include <cmath>
#include <algorithm>

#include "libiqo/AreaResizer.hpp"
#include "libiqo/Types.hpp"

namespace iqo {

    class IAreaResizerImpl
    {
    public:
        //! Destructor
        virtual ~IAreaResizerImpl() {}

        //! Construct impl
        virtual void init(
            size_t srcW, size_t srcH,
            size_t dstW, size_t dstH
        ) = 0;

        //! Run image resizing
        virtual void resize(
            size_t srcSt, const unsigned char * src,
            size_t dstSt, unsigned char * dst
        ) = 0;
    };

    //! MUST NOT define inline function to inherited function.
    //! Because class method inline function will be weak symbol.
    template<class ARCH>
    class AreaResizerImpl : public IAreaResizerImpl
    {
    public:
        virtual ~AreaResizerImpl() {}

        virtual void init(
            size_t srcW, size_t srcH,
            size_t dstW, size_t dstH
        ) {
            (void)srcW;
            (void)srcH;
            (void)dstW;
            (void)dstH;
        }

        virtual void resize(
            size_t srcSt, const unsigned char * src,
            size_t dstSt, unsigned char * dst
        ) {
            (void)srcSt;
            (void)src;
            (void)dstSt;
            (void)dst;
        }
    };

    //! Returns feature capability
    template<class ARCH>
    bool AreaResizerImpl_hasFeature()
    {
        return false;
    }

    //! new AreaResizerImpl<ARCH>
    template<class ARCH>
    IAreaResizerImpl * AreaResizerImpl_new()
    {
        return NULL;
    }

    template<> bool AreaResizerImpl_hasFeature<ArchGeneric>();
    template<> bool AreaResizerImpl_hasFeature<ArchSSE4_1>();
    template<> bool AreaResizerImpl_hasFeature<ArchAVX2FMA>();
    template<> bool AreaResizerImpl_hasFeature<ArchAVX512>();
    template<> bool AreaResizerImpl_hasFeature<ArchNEON>();
    template<> IAreaResizerImpl * AreaResizerImpl_new<ArchGeneric>();
    template<> IAreaResizerImpl * AreaResizerImpl_new<ArchSSE4_1>();
    template<> IAreaResizerImpl * AreaResizerImpl_new<ArchAVX2FMA>();
    template<> IAreaResizerImpl * AreaResizerImpl_new<ArchAVX512>();
    template<> IAreaResizerImpl * AreaResizerImpl_new<ArchNEON>();


    //! Linear integer interpolation
    class LinearIterator
    {
    public:
        LinearIterator(ptrdiff_t dx, ptrdiff_t dy)
        {
            m_DX = dx;
            m_DY = dy;
            m_X = 0;
            m_Y = 0;
        }

        //! set x
        void setX(ptrdiff_t x)
        {
            m_X = (m_DY * x) % m_DX;
            m_Y = (m_DY * x) / m_DX;
        }

        //! get y
        ptrdiff_t operator*() const
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

        ptrdiff_t m_DX;
        ptrdiff_t m_DY;
        ptrdiff_t m_X;
        ptrdiff_t m_Y;
    };

    //! Calculate number of coefficients for area resampling
    size_t calcNumCoefsForArea(size_t srcLen, size_t dstLen);

    //! @brief Set Area table
    //! @param srcLen     Number of pixels of the source image
    //! @param dstLen     Number of pixels of the destination image
    //! @param dstOffset  The coordinate of the destination image
    //! @param numCoefs   Size of table
    //! @param fTable     The table (float or double)
    //! @return Sum of the table
    float setAreaTable(
        size_t srcLen,
        size_t dstLen,
        ptrdiff_t dstOffset,
        ptrdiff_t numCoefs,
        float * fTable
    );

}

namespace {

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

    template<typename T>
    static inline T alignFloor(T v, T alignment)
    {
        return v / alignment * alignment;
    }

    template<typename T>
    static inline T alignCeil(T v, T alignment)
    {
        return (v + (alignment - 1)) / alignment * alignment;
    }

    static inline ptrdiff_t gcd(ptrdiff_t a, ptrdiff_t b)
    {
        ptrdiff_t r = a % b;

        while ( r ) {
            a = b;
            b = r;
            r = a % b;
        }

        return b;
    }

    static inline int64_t lcm(ptrdiff_t a, ptrdiff_t b)
    {
        return int64_t(a) * b / gcd(a, b);
    }

}
