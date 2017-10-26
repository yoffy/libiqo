#pragma once

#include <stdint.h>
#include <cmath>
#include <cstdlib>
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

        //! greatest common divisor
        template<typename T>
        inline T gcd(T a, T b)
        {
            T r = a % b;

            while ( r ) {
                a = b;
                b = r;
                r = a % b;
            }

            return b;
        }

        //! least common multiple
        inline int64_t lcm(ptrdiff_t a, ptrdiff_t b)
        {
            return int64_t(a) / gcd(a, b) * b;
        }

        //! floor(a / b)
        inline int64_t div_floor(int64_t a, int64_t b)
        {
            if ( (a ^ b) < 0 ) {
                return (a - b + 1) / b;
            } else {
                return a / b;
            }
        }

        //! Linear integer interpolation
        //!
        //! Get floor(x * dy/dx) and iterate x.
        class LinearIterator
        {
        public:
            //! @brief constructor
            //! @param dx  denominator
            //! @param dy  numerator
            LinearIterator(ptrdiff_t dx, ptrdiff_t dy)
            {
                m_DX = dx;
                m_DY = dy;
                m_X = 0;
                m_Y = 0;
            }

            //! set x
            //!
            //! y = x * dy/dx
            void setX(ptrdiff_t x)
            {
                m_X = ptrdiff_t((int64_t(x) * m_DY) % m_DX);
                m_Y = ptrdiff_t((int64_t(x) * m_DY) / m_DX);
            }

            //! set x by rational
            //!
            //! y = nume/deno * dy/dx
            void setX(ptrdiff_t nume, ptrdiff_t deno)
            {
                m_Y = ptrdiff_t(div_floor(int64_t(nume)*m_DY, int64_t(deno)*m_DX));

                int64_t newNume = int64_t(nume) * m_DX;
                //int64_t newDeno = int64_t(deno) * m_DX;
                int64_t newDY   = int64_t(m_DY) * deno;
                int64_t newDX   = int64_t(m_DX) * deno;
                int64_t newGCD  = std::abs(gcd(newNume, gcd(newDY, newDX)));
                newNume /= newGCD;
                newDY   /= newGCD;
                newDX   /= newGCD;
                m_X  = ptrdiff_t(newNume % newDX);
                m_X  = (m_X < 0) ? ptrdiff_t(m_X + newDX) : m_X;
                m_DX = ptrdiff_t(newDX);
                m_DY = ptrdiff_t(newDY);
            }

            //! get y
            ptrdiff_t operator*() const
            {
                return m_Y;
            }

            //! ++x
            LinearIterator & operator++()
            {
                advance(1);
                return *this;
            }

            //! x++
            LinearIterator operator++(int)
            {
                LinearIterator tmp(*this);
                advance(1);
                return tmp;
            }

            LinearIterator & operator+=(ptrdiff_t a)
            {
                advance(a);
                return *this;
            }

        private:
            void advance(ptrdiff_t a)
            {
                m_X += a * m_DY;
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

    }
}
