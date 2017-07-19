#pragma once

// Intel

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(i386) || defined(__x86_64__)
    #define IQO_CPU_X86
#endif


#if (defined(IQO_CPU_X86) && (_MSC_VER >= 1500 || __INTEL_COMPILER >= 900)) || defined(__SSE4_1__)
    #define IQO_HAVE_SSE4_1
#endif

#if (defined(IQO_CPU_X86) && (_MSC_VER >= 1600 || __INTEL_COMPILER >= 1100)) || defined(__AVX__)
    #define IQO_HAVE_AVX
#endif

#if (defined(IQO_CPU_X86) && (_MSC_VER >= 1600 || __INTEL_COMPILER >= 1100)) || defined(__FMA__)
    #define IQO_HAVE_FMA
#endif

#if (defined(IQO_CPU_X86) && (_MSC_VER >= 1800 || __INTEL_COMPILER >= 1200)) || (defined(__AVX2__) && defined(__FMA__))
    #define IQO_HAVE_AVX2FMA
#endif

#if (defined(IQO_CPU_X86) && __INTEL_COMPILER >= 1500) || (defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512CD__))
    #define IQO_HAVE_AVX512
#endif


// ARM

#if defined(_M_ARM) || defined(__arm__) || defined(__aarch64__)
    #define IQO_CPU_ARM
#endif

#if defined(__ARM_FEATURE_SIMD32)
    #define IQO_HAVE_ARM_SIMD32
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__ARM_NEON_FP)
    #define IQO_HAVE_NEON
#endif


namespace iqo {

    //! Instruction set for template specialization
    template<int ARCH> struct Arch{};

    enum EnumArch
    {
        kArchGeneric,

        // Intel
        kArchSSE4_1,
        kArchAVX2FMA,
        kArchAVX512,

        // ARM
        //kArchARM_SIMD32,
        kArchNEON,
    };

    typedef Arch<kArchGeneric>  ArchGeneric;

    // Intel
    typedef Arch<kArchSSE4_1>   ArchSSE4_1;     //!< SSE4.1
    typedef Arch<kArchAVX2FMA>  ArchAVX2FMA;    //!< AVX2, FMA
    typedef Arch<kArchAVX512>   ArchAVX512;     //!< AVX512F, AVX512VL, AVX512BW, AVX512DQ, AVX512CD

    // ARM
    //typedef Arch<kArchARM_SIMD32>   ArchARM_SIMD32; //!< SIMD32
    typedef Arch<kArchNEON>         ArchNEON;       //!< NEON

}
