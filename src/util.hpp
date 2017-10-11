#pragma once

#include <cstdlib>

#include "math.hpp"


namespace iqo {
    namespace {

        //! malloc(count * sizeof(T)) and align alignment-bytes
        //!
        //! @param count       Number of elements
        //! @param alignement  Alignment in bytes
        template<typename T>
        T * alignedAlloc(size_t count, size_t alignment)
        {
            // memory layout:
            //
            //   sizeof(void*)+alignment
            //             |
            //     ----------------------
            //     |                    |
            //     | padding | origAddr |..........|
            //     ^         ^          ^
            //     |         |          |
            //   origAddr pOrigAddr  result

            size_t size = count * sizeof(T);
            void * p = std::malloc(size + sizeof(void*) + alignment);
            if ( ! p ) {
                return NULL;
            }

            intptr_t addr = intptr_t(p);
            intptr_t alignedAddr = alignCeil(addr + sizeof(void*), alignment);
            T *      result  = reinterpret_cast<T*>(alignedAddr);
            void **  pOrigAddr = reinterpret_cast<void **>(alignedAddr - sizeof(void*));
            *pOrigAddr = p;
            return result;
        }

        void alignedFree(const void * p)
        {
            if ( ! p ) {
                return;
            }
            intptr_t alignedAddr = intptr_t(p);
            void **  pOrigAddr = reinterpret_cast<void **>(alignedAddr - sizeof(void*));
            std::free(*pOrigAddr);
        }

    }
}