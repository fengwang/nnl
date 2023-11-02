#ifndef RJOGIGPHDGPCKKWRIRPGHJWNOEVEOKDFQPQWXCKVNYOUONEXSBJDYNTLSYOKHNVHSPNOUJNJX
#define RJOGIGPHDGPCKKWRIRPGHJWNOEVEOKDFQPQWXCKVNYOUONEXSBJDYNTLSYOKHNVHSPNOUJNJX

#include "../utility/utility.hpp"
#include "./session_context.hpp"
#include "./engine.hpp"
#include "./stream.hpp"
#include "./allocator.hpp"

extern "C" void cuda_device_synchronize(); // impls in src/cuda/device_management.cpp

namespace nnl
{

    template< Engine E >
    struct session;

    template< Engine E >
    session<E>& get_default_session();

    template< Engine E >
    void host2device( std::byte* src, std::size_t n, std::byte* dst, stream<E>& e );
    template< Engine E >
    void device2host( std::byte* src, std::size_t n, std::byte* dst, stream<E>& e );
    template< Engine E >
    void device2device( std::byte* src, std::size_t n, std::byte* dst, stream<E>& e );

    void host2device( std::byte* src, std::size_t n, std::byte* dst, stream<cuda_engine>& e );
    void device2host( std::byte* src, std::size_t n, std::byte* dst, stream<cuda_engine>& e );
    void device2device( std::byte* src, std::size_t n, std::byte* dst, stream<cuda_engine>& e );

    template< Engine E >
    std::tuple<std::int64_t, std::byte*> allocate_maximum_device_memory();

    template<>
    std::tuple<std::int64_t, std::byte*> allocate_maximum_device_memory<cuda_engine>();

    template< Engine E >
    void reset_device();

    template<>
    void reset_device<cuda_engine>(); // <- defined in memory_management.cpp

    template< Engine E >
    struct device_memory_manager
    {
        device_memory max_device_memory_;

        std::set<device_memory> memory_blocks_; // all splitted device memory blocks
        std::set<device_memory> assigned_memory_blocks_; // all assigned device memory blocks
        std::multimap<std::int64_t, device_memory> assigned_memory_; // allocated, assigned
        std::multimap<std::int64_t, device_memory> idle_memory_; // allocated, assigned and then dismissed memory, <size-mem>

        host_allocator<E, std::byte> host_allocator_;
        device_allocator<E, std::byte> device_allocator_;

        stream<E> default_io_stream_;
        stream<E> default_arithmetic_stream_;
        stream<E> default_host2device_stream_;
        stream<E> default_device2host_stream_;
        stream<E> default_weight_preloading_stream_;

        stream<E>& default_host2device_stream()
        {
            return default_host2device_stream_;
        }

        stream<E>& default_device2host_stream()
        {
            return default_device2host_stream_;
        }

        stream<E>& default_weight_preloading_stream()
        {
            return default_weight_preloading_stream_;
        }

        device_memory_manager()
        {
            // TODO: select fastest device on initialize
            max_device_memory_.address_ = nullptr;
            max_device_memory_.size_in_bytes_ = 0;
        }

        //
        // recall all assigned memory, only max_device_memory_ and idle_memory has values
        //
        void clear()
        {
            memory_blocks_.clear();
            assigned_memory_blocks_.clear();
            assigned_memory_.clear();
            idle_memory_.clear();
            {
                auto mem = max_device_memory_.address_;
                auto size = max_device_memory_.size_in_bytes_;
                idle_memory_.insert( std::make_pair( size, device_memory{mem, size} ) );
            }
        }

        ~device_memory_manager()
        {
            // TODO: device reset // cudaDeviceReset()
            device_allocator_.deallocate( max_device_memory_.address_ );
            max_device_memory_.address_ = nullptr;
            max_device_memory_.size_in_bytes_ = 0;
            memory_blocks_.clear();
            assigned_memory_blocks_.clear();
            assigned_memory_.clear();
            idle_memory_.clear();

            synchronize();
        }

        device_memory acquire_memory( std::int64_t size_in_bytes )
        {
            // dirty hack
            size_in_bytes = ((size_in_bytes >> 4) + 1) << 4;

            // acquire maximum_memory on first allocation
            if (max_device_memory_.address_ == nullptr )
            {
                auto [size, mem] = allocate_maximum_device_memory<E>(); // <- can this be misaligned?

                max_device_memory_.address_ = mem;
                max_device_memory_.size_in_bytes_ = size;

                // TODO: everything below should be empty, double check here just in case
                idle_memory_.clear();
                memory_blocks_.clear();
                assigned_memory_blocks_.clear();
                assigned_memory_.clear();
                idle_memory_.insert( std::make_pair( size, device_memory{mem, size} ) );
            }

            // if possible from idle memory
            if ( auto it = idle_memory_.find( size_in_bytes ); it != idle_memory_.end() )
            {
                device_memory mem = it->second;
                assigned_memory_.insert( std::make_pair(size_in_bytes, mem) );
                idle_memory_.erase( it );
                assigned_memory_blocks_.insert( mem );
                return mem;
            }

            // no available memory, i.e., even the largest idle memory block is too small
            if ( (idle_memory_.rbegin())->first < size_in_bytes )
            {
                spdlog::info( "acquire_memory:: align_assigned_memory triggered when acquiring {} bytes memory.", size_in_bytes );
                align_assigned_memory();//merge small blocks
            }

            // split the max memory block, which is the last element in the idle_memory_
            if ( (idle_memory_.rbegin())->first > size_in_bytes )
            {
                auto [max_size_in_bytes, mem] = *(idle_memory_.rbegin());
                auto [memory_address, memory_size] = mem;

                std::int64_t const new_memory_size = size_in_bytes;
                std::byte* new_memory_address = memory_address;
                device_memory new_memory{ new_memory_address, new_memory_size };

                std::int64_t const rest_memory_size = max_size_in_bytes - size_in_bytes;
                std::byte* rest_memory_address = memory_address + size_in_bytes;
                device_memory rest_memory{ rest_memory_address, rest_memory_size };

                //updated memory_blocks_
                {
                    memory_blocks_.erase( mem );
                    memory_blocks_.insert( new_memory );
                    memory_blocks_.insert( rest_memory );
                }

                //update idle_memory_
                {
                    //idle_memory_.erase( --idle_memory_.end() );
                    idle_memory_.erase( std::prev(idle_memory_.end()) );
                    //insert rest
                    idle_memory_.insert( std::make_pair(rest_memory_size, rest_memory) );
                }

                //update assigned_memory_
                {
                    //insert new
                    assigned_memory_.insert( std::make_pair(size_in_bytes, new_memory) );
                    assigned_memory_blocks_.insert( new_memory );
                }

                return new_memory;
            }

            spdlog::error( nnl::format( "device_memory_manager::acquire_memory: Error failed to allocate {} bytes memory.\nCurrent memory manager status: {}", size_in_bytes, (*this) ) );
            //better_assert( false, nnl::format("Failed to allocate {} bytes memory.\nCurrent status:\n{}", size_in_bytes, (*this)) );
            return {};
        }

        // TODO:
        // should have add_idle/add_assigned/delete_idle/delete_assigned methods defined
        //
        void dismiss_memory( device_memory const& mem )
        {
            if constexpr (debug_mode)
            {
                spdlog::info( "dismissing memory: {0:d} byte at [{1:x}, {1:x}].", mem.size(), (std::int64_t)mem.data(), ((std::int64_t)mem.data())+mem.size() );
            }

            if ( memory_blocks_.find(mem) == memory_blocks_.end() )
            {
                spdlog::error( "Failed to dismiss device memory {0:d}:{1:x}. Not found from memory blocks.", mem.size(), (std::int64_t)mem.data() );
                better_assert( false );
                return;
            }

            // <- to keep things runing
            if constexpr (debug_mode)
            {
                //if ( mem.size() < 1024UL*1024 )
                {
                    std::uint64_t const mx_size_to_dump = std::min( 1024UL*1024, (mem.size() >> 2) + 1UL );
                    std::vector<std::float32_t, host_allocator<default_engine_type, std::float32_t>> cache;
                    cache.resize( mx_size_to_dump );

                    //device2host( mem.data(), mx_size_to_dump, reinterpret_cast<std::byte*>(cache.data()), default_io_stream_ );
                    //device2host( mem.data(), mx_size_to_dump, reinterpret_cast<std::byte*>(cache.data()), default_device2host_stream_ );
                    device2host( mem.data(), (mx_size_to_dump<<2), reinterpret_cast<std::byte*>(cache.data()), default_device2host_stream_ );

#if 1
                    // DOING test here
                    //default_io_stream_.synchronize(); // <-- must sync for a async memcpy
                    default_device2host_stream_.synchronize(); // <-- must sync for a async memcpy
                    std::chrono::time_point<std::chrono::system_clock> const& now = std::chrono::system_clock::now();
                    std::string const& file_name = nnl::format( "./{}_dismissed_memory_size_{}_address_{}.txt", now.time_since_epoch(), mem.size(), (std::int64_t)mem.data() );
                    std::ofstream ofs{ file_name };
                    std::copy( cache.begin(), cache.end(), std::ostream_iterator<float>{ ofs, "\t" } );
                    ofs.close();

                    spdlog::info( "dismiss_memory: dump a memory block with size {} starting at {}  to file {}.", mem.size(), (std::int64_t)mem.data(), file_name );
#endif
                }
            }


            std::byte* next_block_data_address = mem.data()+mem.size();
            for ( auto itor = idle_memory_.begin(); itor != idle_memory_.end(); ++itor ) //TODO: optimization here?
            {
                std::int64_t next_size = itor->first;
                device_memory dm = itor->second;
                if (dm.data() == next_block_data_address) // next block is also empty
                {
                    // delete next block
                    memory_blocks_.erase( dm );
                    idle_memory_.erase( itor );
                    // delete mem
                    memory_blocks_.erase( mem );
                    assigned_memory_blocks_.erase( mem );
                    auto [address, size] = mem;
                    auto rg = assigned_memory_.equal_range(size);
                    for ( auto it = rg.first; it != rg.second; ++it )
                    {
                        if (it->second == mem )
                        {
                            assigned_memory_.erase( it );
                            break;
                        }
                    }
                    // create a fake assigned new_mem
                    device_memory new_mem{ mem.data(), mem.size()+next_size };
                    spdlog::info( "creating a new device memory with own size {} and next size {}", mem.size(), next_size);
                    memory_blocks_.insert( new_mem );
                    assigned_memory_blocks_.insert( new_mem );
                    assigned_memory_.insert( std::make_pair( new_mem.size(), new_mem ) );
                    //
                    return dismiss_memory( new_mem );
                }
            }

            //remove it from assigned memory
            auto [address, size] = mem;
            auto rg = assigned_memory_.equal_range(size);
            for ( auto it = rg.first; it != rg.second; ++it )
            {
                if (it->second == mem )
                {
                    assigned_memory_.erase( it );
                    break;
                }
            }
            //insert it to idle memory
            idle_memory_.insert( std::make_pair( size, mem ) );

            // udpated assigned_memory_blocks_
            if ( 1 != assigned_memory_blocks_.erase( mem ) )
                spdlog::error( "Failed to remove memory ({0:x}, {1:d}) from assigned_memory_blocks_", (std::int64_t)mem.address_, mem.size_in_bytes_ );

            // try to merge with the prev idle memory block
            //std::multimap<std::int64_t, device_memory> idle_memory_; // allocated, assigned and then dismissed memory, <size-mem>
            for ( auto [_, dm] : idle_memory_ )
            {
                if ( dm.data() + dm.size() == mem.data() )
                {
                    device_memory new_dm{ dm.data(), dm.size()+mem.size() };
                    // remove prev
                    {
                        bool prev_removed = false;
                        auto rg = idle_memory_.equal_range( dm.size() );
                        for ( auto itor : range( rg.first, rg.second) )
                        {
                            if ( (itor->second).data() == dm.data() )
                            {
                                idle_memory_.erase( itor );
                                prev_removed = true;
                                break;
                            }
                        }
                        if ( !prev_removed )
                        {
                            spdlog::error( format( "Failed to remove prev memblock from idle_memory: {}", dm ) );
                            std::abort();
                        }

                        if ( 1 != memory_blocks_.erase( dm ) )
                        {
                            spdlog::error( format( "Failed to remove prev memblock from memory_blocks_: {}", dm ) );
                            std::abort();
                        }
                    }
                    // remove current mem
                    {
                        bool current_removed = false;
                        auto rg = idle_memory_.equal_range( mem.size() );
                        for ( auto itor : range( rg.first, rg.second) )
                        {
                            if ( (itor->second).data() == mem.data() )
                            {
                                idle_memory_.erase( itor );
                                current_removed = true;
                                break;
                            }
                        }
                        if ( !current_removed )
                        {
                            spdlog::error( format( "Failed to remove memblock from idle_memory: {}", mem ) );
                            std::abort();
                        }
                        if ( 1 != memory_blocks_.erase( mem ) )
                        {
                            spdlog::error( format( "Failed to remove prev memblock from memory_blocks_: {}", mem ) );
                            std::abort();
                        }
                    }

                    idle_memory_.insert( std::make_pair( new_dm.size(), new_dm ) ); // <- insert the merged memory
                    memory_blocks_.insert( new_dm );

                    return;
                }
            }

        }

        void align_assigned_memory()
        {
            if constexpr ( debug_mode )
            {
                std::cout << "Before aligning memory:\n" << *this << std::endl;
            }

            auto& sess = get_default_session<E>();

            std::set<device_memory> new_memory_blocks_; // all splitted device memory blocks
            std::set<device_memory> new_assigned_memory_blocks_; // all assigned device memory blocks
            std::multimap<std::int64_t, device_memory> new_assigned_memory_; // allocated, assigned
            std::multimap<std::int64_t, device_memory> new_idle_memory_; // allocated, assigned and then dismissed memory

            //std::byte* start_address = maximum_memory_.address_;
            std::byte* start_address = max_device_memory_.address_;
            std::int64_t assigned_memory_in_bytes = 0;
            for ( auto& old_memory : assigned_memory_blocks_ )
            {
                device_memory new_memory{ start_address, old_memory.size_in_bytes_ };
                start_address += old_memory.size_in_bytes_;
                assigned_memory_in_bytes += old_memory.size_in_bytes_;

                new_memory_blocks_.insert( new_memory );
                new_assigned_memory_blocks_.insert( new_memory );
                new_assigned_memory_.insert( std::make_pair( old_memory.size_in_bytes_, new_memory ) );
                sess.update_device_memory( old_memory, new_memory );

                if ( old_memory != new_memory ) [[unlikely]]
                {
                    device2device_copy( old_memory.address_, old_memory.address_+old_memory.size_in_bytes_, new_memory.address_, default_io_stream_ );
                    spdlog::info( format( "align_assigned_memory:: copying old memory {} to new memory {}", old_memory, new_memory ) );
                }
            }
            new_idle_memory_.insert( std::make_pair( max_device_memory_.size_in_bytes_-assigned_memory_in_bytes, device_memory{start_address, max_device_memory_.size_in_bytes_-assigned_memory_in_bytes} ) );

            // update all indices
            {
                std::swap( memory_blocks_, new_memory_blocks_ );
                std::swap( assigned_memory_blocks_, new_assigned_memory_blocks_ );
                std::swap( assigned_memory_, new_assigned_memory_ );
                std::swap( idle_memory_, new_idle_memory_ );
            }

            if constexpr ( debug_mode )
            {
                std::cout << "After aligning memory:\n" << *this << std::endl;
            }
        }

        device_memory_manager( device_memory_manager const& ) = default;
        device_memory_manager( device_memory_manager && ) = default;
        device_memory_manager& operator=( device_memory_manager const& ) =  default;
        device_memory_manager& operator=( device_memory_manager && ) =  default;

        void device2host_copy_n( std::byte* src, std::int64_t n, std::byte* dst, stream<E>& sm )
        {
            device2host_copy( src, src+n, dst, sm );
        }

        void host2device_copy_n( std::byte* src, std::int64_t n, std::byte* dst, stream<E>& sm )
        {
            host2device_copy( src, src+n, dst, sm );
        }

        void device2device_copy_n( std::byte* src, std::int64_t n, std::byte* dst, stream<E>& sm )
        {
            device2device_copy( src, src+n, dst, sm );
        }

        std::byte* host_allocate( std::int64_t n )
        {
            return host_allocator_.allocate( n );
        }

        void host_deallocate( std::byte* address, std::int64_t n )
        {

            return host_allocator_.deallocate( address, n );
        }

        stream<E>& default_stream()
        {
            return default_io_stream_;
        }

        stream<E>& default_io_stream()
        {
            return default_io_stream_;
        }

        stream<E>& default_arithmetic_stream()
        {
            return default_arithmetic_stream_;
        }

        stream<E> const& default_stream() const
        {
            return default_io_stream_;
        }

        stream<E> const& default_io_stream() const
        {
            return default_io_stream_;
        }

        stream<E> const& default_arithmetic_stream() const
        {
            return default_arithmetic_stream_;
        }

        void synchronize()
        {
            cuda_device_synchronize();
            /*
            default_io_stream_.synchronize();
            default_arithmetic_stream_.synchronize();
            default_host2device_stream_.synchronize();
            default_device2host_stream_.synchronize();
            default_weight_preloading_stream_.synchronize();
            */

            //spdlog::debug( "default streams are synchronized." );
        }

    private:
        void host2device_copy( std::byte* first, std::byte* last, std::byte* dest, stream<E>& sm )
        {
            host2device( first, last-first, dest, sm );
        }

        void device2host_copy( std::byte* first, std::byte* last, std::byte* dest, stream<E>& sm )
        {
            device2host( first, last-first, dest, sm );
        }

        void device2device_copy( std::byte* first, std::byte* last, std::byte* dest, stream<E>& sm )
        {
            device2device( first, last-first, dest, sm );
        }
    };//device_memory_manager

    template< Engine E >
    std::ostream& operator<<( std::ostream& os, device_memory_manager<E> const& dmm )
    {
        os << "Device Memory Manager:\n";
        os << "\tMax Device Memory:\n";
        os << "\t\t" << dmm.max_device_memory_ << "\n";
        os << "\tMemory Blocks:\n";
        for ( auto const& dm : dmm.memory_blocks_ )
            os << "\t\t" << dm << "\n";
        os << "\tAssigned Memory Blocks:\n";
        for ( auto const& dm : dmm.assigned_memory_blocks_ )
            os << "\t\t" << dm << "\n";
        os << "\tAssigned Memory:\n";
        for ( auto const& [size, dm] : dmm.assigned_memory_ )
            os << "\t\t[" << size << ", " << dm << "]\n";
        os << "\tIdle Memory:\n";
        for ( auto const& [size, dm] : dmm.idle_memory_ )
            os << "\t\t[" << size << ", " << dm << "]\n";
        os << "\tIO Stream: " << dmm.default_io_stream() << "\n";
        os << "\tArithmetic Stream: " << dmm.default_arithmetic_stream() << "\n";

        return os;
    }

    template< Engine E > // <-- template E to postpone instancing
    struct device_memory_buffer
    {
        device_memory  device_memory_buffer_;

        device_memory_buffer()
        {
            device_memory_buffer_ = device_memory{ nullptr, 0 };
        }

        void reserve( std::int64_t size_in_bytes )
        {
            better_assert( 0 == device_memory_buffer_.size(), "cannot reserve device memory buffer more than once." );
            device_memory_buffer_ = get_default_session<E>().dm().acquire_memory( size_in_bytes );
        }

        std::byte* data() const
        {
            return device_memory_buffer_.data();
        }

        std::int64_t size() const
        {
            return device_memory_buffer_.size();
        }

        ~device_memory_buffer()
        {
#if 0
            if ( device_memory_buffer_.size() )
            {
                spdlog::info( format("device_memory_buffer {} deallocating", device_memory_buffer_) );
                get_default_session<E>().dm().dismiss_memory( device_memory_buffer_ );
                device_memory_buffer_ = device_memory{nullptr, 0};
            }
#endif
        }
    }; // struct device_memory_buffer





}//namespace nnl

#endif//RJOGIGPHDGPCKKWRIRPGHJWNOEVEOKDFQPQWXCKVNYOUONEXSBJDYNTLSYOKHNVHSPNOUJNJX

