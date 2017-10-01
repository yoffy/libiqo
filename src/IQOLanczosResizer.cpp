#include "IQOLanczosResizerImpl.hpp"
#include "IQOHWCap.hpp"


namespace iqo {

    LanczosResizer::LanczosResizer(
        unsigned int degree,
        size_t srcW,
        size_t srcH,
        size_t dstW,
        size_t dstH,
        size_t pxScale
    ) {
        HWCap cap;

#if defined(IQO_CPU_X86)
        if ( cap.hasAVX512() && (m_Impl = LanczosResizerImpl_new<ArchAVX512>()) ) {
            goto L_found;
        }
        if ( cap.hasAVX2FMA() && (m_Impl = LanczosResizerImpl_new<ArchAVX2FMA>()) ) {
            goto L_found;
        }
        if ( cap.hasSSE4_1() && (m_Impl = LanczosResizerImpl_new<ArchSSE4_1>()) ) {
            goto L_found;
        }
#elif defined(IQO_CPU_ARM)
        if ( cap.hasNEON() && (m_Impl = LanczosResizerImpl_new<ArchNEON>()) ) {
            goto L_found;
        }
#endif

        m_Impl = LanczosResizerImpl_new<ArchGeneric>();

L_found:
        m_Impl->init(degree, srcW, srcH, dstW, dstH, pxScale);
    }

    LanczosResizer::~LanczosResizer()
    {
        delete m_Impl;
    }

    void LanczosResizer::resize(
        size_t srcSt, const unsigned char * src,
        size_t dstSt, unsigned char * dst
    ) {
        m_Impl->resize(srcSt, src, dstSt, dst);
    }

}
