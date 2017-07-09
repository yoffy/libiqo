#include "IQOHWCap.hpp"

namespace iqo {

    HWCap::HWCap()
    {
        #if defined(IQO_CPU_X86) && (defined(__GNUC__) || defined(__clang__))
            cpuid(0x01, 0x00, m_0x01_0x00);
            cpuid(0x07, 0x00, m_0x07_0x00);
        #endif
    }

    #if defined(IQO_CPU_X86) && (defined(__GNUC__) || defined(__clang__))
        void HWCap::cpuid(unsigned int eax, unsigned int ecx, CPUID & dst)
        {
            __asm__ volatile(
                "cpuid"
                : "=a"(dst.a), "=b"(dst.b), "=c"(dst.c), "=d"(dst.d)
                : "a"(eax), "c"(ecx)
            );
        }
    #endif

}
