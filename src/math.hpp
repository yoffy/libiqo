#pragma once

#include <stdint.h>
#include <cmath>
#include <algorithm>


namespace iqo {
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
        inline T alignFloor(T v, T alignment)
        {
            return v / alignment * alignment;
        }

        template<typename T>
        inline T alignCeil(T v, T alignment)
        {
            return (v + (alignment - 1)) / alignment * alignment;
        }

        inline ptrdiff_t gcd(ptrdiff_t a, ptrdiff_t b)
        {
            ptrdiff_t r = a % b;

            while ( r ) {
                a = b;
                b = r;
                r = a % b;
            }

            return b;
        }

        inline int64_t lcm(ptrdiff_t a, ptrdiff_t b)
        {
            return int64_t(a) * b / gcd(a, b);
        }

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
        size_t calcNumCoefsForLinear(size_t srcLen, size_t dstLen);

        //! @brief Set Linear table
        //! @param srcLen     Number of pixels of the source image
        //! @param dstLen     Number of pixels of the destination image
        //! @param dstOffset  The coordinate of the destination image
        //! @param numCoefs   Size of table
        //! @param fTable     The table (float or double)
        //! @return Sum of the table
        float setLinearTable(
            size_t srcLen,
            size_t dstLen,
            ptrdiff_t dstOffset,
            ptrdiff_t numCoefs,
            float * fTable
        );

    }
}
