#include <libiqo/Types.hpp>

namespace iqo {

    //! Hardware Capability
    class HWCap
    {
    public:
        static int getNumberOfProcs();
        static int getThreadNumber();

        HWCap();

#if defined(IQO_CPU_X86)
        bool hasSSE4_1() const      { return m_0x01_0x00.c & (1 << 19); }
        bool hasSSE4_2() const      { return m_0x01_0x00.c & (1 << 20); }
        bool hasAVX() const         { return m_0x01_0x00.c & (1 << 28); }
        bool hasAVX2() const        { return m_0x07_0x00.b & (1 <<  5); }
        bool hasFMA() const         { return m_0x01_0x00.c & (1 << 12); }
        bool hasAVX512F() const     { return m_0x07_0x00.b & (1 << 16); }
        bool hasAVX512VL() const    { return m_0x07_0x00.b & (1 << 31); }
        bool hasAVX512BW() const    { return m_0x07_0x00.b & (1 << 30); }
        bool hasAVX512DQ() const    { return m_0x07_0x00.b & (1 << 17); }
        bool hasAVX512CD() const    { return m_0x07_0x00.b & (1 << 28); }
        bool hasBMI1() const        { return m_0x07_0x00.b & (1 <<  3); }
        bool hasBMI2() const        { return m_0x07_0x00.b & (1 <<  8); }

        bool hasAVX2FMA() const;    //!< AVX2 & FMA
        bool hasAVX512() const;     //!< F && VL && BW && DQ && CD
#endif

#if defined(IQO_CPU_ARM)
    #if defined(IQO_HAVE_NEON)
        bool hasNEON() const        { return true; }
    #else
        bool hasNEON() const        { return false; }
    #endif
#endif

    private:
        // no copy
        HWCap(const HWCap &);
        HWCap & operator=(const HWCap &);

#if defined(IQO_CPU_X86)
        union CPUID
        {
            struct { unsigned int a, b, c, d; };
            int info[4];
        };

        CPUID m_0x01_0x00;
        CPUID m_0x07_0x00;

        static void cpuid(unsigned int eax, unsigned int ecx, CPUID & dst);
#endif
    };

}
