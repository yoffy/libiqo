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

#if defined(IQO_CPU_X86) && defined(IQO_HAVE_AVX2FMA)
        //! Linear integer interpolation
        //!
        //! Get floor(x * dy/dx) and iterate x.
        class LinearIteratorAVX2
        {
        public:
            enum {
                kNumElems = sizeof(__m256i) / sizeof(int32_t)
            };

            //! @brief constructor
            //! @param dx  denominator
            //! @param dy  numerator
            LinearIteratorAVX2(int32_t dx, int32_t dy)
            {
                const __m256 s32x8kCols = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
                m_DX = _mm256_set1_epi32(dx);
                m_DY = _mm256_set1_epi32(dy);
                m_X = _mm256_mullo_epi32(s32x8kCols, m_DY);
                m_Y = _mm256_setzero_si256();
                advance(0);
            }

            //! get y
            __m256i operator*() const
            {
                return m_Y;
            }

            //! ++x
            LinearIteratorAVX2 & operator++()
            {
                advance(1);
                return *this;
            }

            //! x++
            LinearIteratorAVX2 operator++(int)
            {
                LinearIteratorAVX2 tmp(*this);
                advance(1);
                return tmp;
            }

            LinearIteratorAVX2 & operator+=(int32_t a)
            {
                advance(a);
                return *this;
            }

        private:
            //! returns a >= b
            static __m256i cmpge_epi32(__m256i a, __m256i b)
            {
                __m256i vFF = _mm256_set1_epi8(0xFF);
                return _mm256_xor_si256(vFF, _mm256_cmpgt_epi32(b, a));
            }

            void advance(int32_t a)
            {
                __m256i s32x8A = _mm256_set1_epi32(a * kNumElems);

                // m_X += a * m_DY;
                m_X = _mm256_add_epi32(m_X, _mm256_mullo_epi32(s32x8A, m_DY));
                // isGE = m_X >= m_DX;
                __m256i isGE = cmpge_epi32(m_X, m_DX);
                while ( ! _mm256_testz_si256(isGE, isGE) ) {
                    // if ( isGE[i] ) m_Y[i]++;
                    m_Y = _mm256_sub_epi32(m_Y, isGE);
                    // if ( isGE[i] ) m_X[i] -= m_DX[i];
                    m_X = _mm256_sub_epi32(m_X, _mm256_and_si256(m_DX, isGE));

                    // isGE = m_X >= m_DX;
                    isGE = cmpge_epi32(m_X, m_DX);
                }
            }

            __m256i m_DX;
            __m256i m_DY;
            __m256i m_X;
            __m256i m_Y;
        };
#endif

    }
}
