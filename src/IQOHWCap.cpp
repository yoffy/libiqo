#include "IQOHWCap.hpp"

#if defined(_MSC_VER)
    #include <intrin.h>
#endif


namespace iqo {

    HWCap::HWCap()
    {
#if defined(IQO_CPU_X86)
        cpuid(0x01, 0x00, m_0x01_0x00);
        cpuid(0x07, 0x00, m_0x07_0x00);
#endif
    }

#if defined(IQO_CPU_X86)
    void HWCap::cpuid(unsigned int eax, unsigned int ecx, CPUID & dst)
    {
    #if defined(__GNUC__) || defined(__clang__)
        __asm__ volatile(
            "cpuid"
            : "=a"(dst.a), "=b"(dst.b), "=c"(dst.c), "=d"(dst.d)
            : "a"(eax), "c"(ecx)
        );
    #elif defined(_MSC_VER)
        __cpuidex(dst.info, eax, ecx);
    #endif
    }
#endif

}
