#ifndef CACHED_ALLOCATOR_HPP_INCLUDED_FOPADIJASLDKJASLDFKJASDKLJFOIJFSAI84LKAFJF
#define CACHED_ALLOCATOR_HPP_INCLUDED_FOPADIJASLDKJASLDFKJASDKLJFOIJFSAI84LKAFJF

#include "../include.hpp"
#include "../config.hpp"
#include "./singleton.hpp"

namespace nnl
{

    //---------------------------------------------------------
    //
    // cached_allocator is designed for Host Tensor only.
    //
    //---------------------------------------------------------


    namespace
    {
        struct memory
        {
            unsigned long size_;
            std::byte* address_;
            friend constexpr auto operator <=> ( memory const&, memory const& ) noexcept = default;
        }; // struct memory


        struct memory_hash
        {
            unsigned long operator()( memory const& mem ) const noexcept
            {
                return ( std::hash<unsigned long>()( mem.size_ ) << 1 ) ^ std::hash<std::byte*>()( mem.address_ );
            }
        }; // struct memory_hash

        template< typename Host_Allocator >
        struct memory_cache;

        template< typename Host_Allocator >
        std::ostream& operator << ( std::ostream& os,  memory_cache<Host_Allocator> const& mc )
        {
            os << "Memory Cache:\n";
            os << "  Allocated Memory:\n";
            for ( auto const& mem : mc.allocated_memory )
                os << "    + " << mem.size_ << "->" << (std::int64_t)mem.address_ << "\n";
            os << "  Reserved Memory:\n";
            for ( auto const& [size, address] : mc.reserved_memory )
                os << "    + " << size << "->" << (std::int64_t)address << "\n";
            return os;
        }

        template< typename Host_Allocator >
        struct memory_cache
        {
            std::unordered_set<memory, memory_hash> allocated_memory;
            std::unordered_multimap<unsigned long, std::byte*> reserved_memory;
            Host_Allocator alloc;

            std::byte* allocate( unsigned long size )
            {
                // search reserved_memory, return if found one matches
                auto search = reserved_memory.find( size );
                if ( search != reserved_memory.end() )
                {
                    std::byte* ans = search->second;
                    allocated_memory.emplace( size, ans );
                    reserved_memory.erase( search );

                    std::cout << "After allocating " << size << " bytes of memory, the memory cache is updated to:\n" << (*this) << std::endl;

                    return ans;
                }

                std::byte* ans = nullptr;
                try
                {
                    ans = alloc.allocate( size );
                }
                catch (const std::bad_alloc& e)
                {
                    if ( gc() ) // another attempt to allocate memory
                    {
                        ans = alloc.allocate( size );
                    }
                    else
                    {
                        std::abort();
                    }
                }

                allocated_memory.emplace( size, ans );

                std::cout << "After allocating " << size << " bytes of memory, the memory cache is updated to:\n" << (*this) << std::endl;

                return ans;
            }

            void deallocate( std::byte* ptr, unsigned long size )
            {
                allocated_memory.erase( memory{size, ptr} );
                reserved_memory.emplace( size, ptr );

                std::cout << "After deallocating " << size << " bytes of memory at " << (std::int64_t)ptr
                          << ", the memory cache is updated to:\n" << (*this) << std::endl;
            }

            ~memory_cache()
            {
                std::cout << "Before destroying memory cache, the snapshot is:\n" << (*this) << std::endl;

                for ( auto& mem : allocated_memory )
                {
                    auto& [size, address] = mem;
                    alloc.deallocate( address, size );
                }

                for ( auto& [size, address] : reserved_memory )
                {
                    alloc.deallocate( address, size );
                }

                allocated_memory.clear();
                reserved_memory.clear();
            }

            // garbage collection, true for success, false for failure.
            bool gc()
            {
                std::cout << "Before GC memory cache, the snapshot is:\n" << (*this) << std::endl;

                if (reserved_memory.size()==0)
                    return false;

                for ( auto& [size, address] : reserved_memory )
                    alloc.deallocate( address, size );

                reserved_memory.clear();

                return true;
            }

        }; // struct memory_cache

        template< typename Host_Allocator>
        memory_cache<Host_Allocator>& get_memory_cache()
        {
            return singleton<memory_cache<Host_Allocator>>::instance();
        }
    }//anonymous namespace

    //
    // Warning: only for tensor
    //
    template< typename T, typename Host_Allocator=std::allocator<std::byte>> requires (not std::same_as<T, void>)
    struct cached_allocator
    {
        typedef T value_type;
        typedef unsigned long size_type;
        typedef std::ptrdiff_t difference_type;

        [[nodiscard]] T* allocate( unsigned long const n )
        {
            std::byte* ans = get_memory_cache<Host_Allocator>().allocate( n*sizeof(T) );
            std::memset( ans, 0, n*sizeof(T) );
            return reinterpret_cast<T*>( ans );
        }

        void deallocate( T* p, unsigned long const n )
        {
            get_memory_cache<Host_Allocator>().deallocate( reinterpret_cast<std::byte*>(p), n * sizeof(T) );
        }

        // empty construct for performance reasons.
        template< typename U, typename ... Args >
        void construct( U*, Args&& ... )
        {
        }

        // empty construct_at for performance optimization reasons.
        template< typename U, typename ... Args>
        constexpr U* construct_at( U* ptr, Args&&... )
        {
            return ptr;
        }

        // empty destroy for performance reasons.
        template< typename U >
        void destroy( U* )
        {
        }

        //althought this has been removed in std::allocator from C++20 on, some STL's allocator_trait still relies on this embeded class
        template< class U > struct rebind { typedef cached_allocator<U, Host_Allocator> other; };
    };

    template< class T1, class T2, typename Host_Allocator >
    constexpr bool operator==( const cached_allocator<T1, Host_Allocator>&, const cached_allocator<T2, Host_Allocator>& ) noexcept
    {
        return true;
    }

}//nnl

#endif//CACHED_ALLOCATOR_HPP_INCLUDED_FOPADIJASLDKJASLDFKJASDKLJFOIJFSAI84LKAFJF

