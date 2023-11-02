#ifndef PBBMHSHQVGLAHLTKNTAUXYOGSLYGYOBKVYXJTDTCCHERVYPCHOUJEECFFCOWXDWTDDVPPUULL
#define PBBMHSHQVGLAHLTKNTAUXYOGSLYGYOBKVYXJTDTCCHERVYPCHOUJEECFFCOWXDWTDDVPPUULL

#include "../utility/utility.hpp"

namespace nnl
{
    struct device_memory
    {
        std::byte* address_;
        std::int64_t size_in_bytes_;

        device_memory( std::byte* address=nullptr, std::int64_t size_in_bytes=0 ) : address_{address}, size_in_bytes_{size_in_bytes} {}

        friend auto operator <=> ( device_memory const&, device_memory const& ) = default;

        std::int64_t size() const
        {
            return size_in_bytes_;
        }

        std::byte* data() const
        {
            return address_;
        }
    };//device_memory




    inline std::ostream& operator<< ( std::ostream& os, device_memory const& dm )
    {
        //return os << std::hex << "[" << dm.address_ << ", " << std::dec << dm.size_in_bytes_ << "]";
        if ( dm.size_in_bytes_ < 1024 )
            return os << std::hex << "[" << dm.address_ << " ->" << std::dec << dm.size_in_bytes_ << " Bytes <- " << std::hex << dm.address_+dm.size_in_bytes_ << "]";

        if ( dm.size_in_bytes_ < 1024*1024 )
            return os << std::hex << "[" << dm.address_ << " ->" << std::dec << (dm.size_in_bytes_>>10) << "K <- " << std::hex << dm.address_+dm.size_in_bytes_ << "]";

        if ( dm.size_in_bytes_ < 1024*1024*1024 )
            return os << std::hex << "[" << dm.address_ << " ->" << std::dec << (dm.size_in_bytes_>>20) << "M <- " << std::hex << dm.address_+dm.size_in_bytes_ << "]";

        return os << std::hex << "[" << dm.address_ << " ->" << std::dec << (dm.size_in_bytes_>>30) << "G <- " << std::hex << dm.address_+dm.size_in_bytes_ << "]";
    }

    struct record
    {
        std::int64_t    tensor_id_;
        std::byte*      device_memory_;
        std::int64_t    memory_size_in_bytes_;
    };//struct record

    inline std::ostream& operator<< ( std::ostream& os, record const& r )
    {
        return os << "[(" << r.tensor_id_ << ") " << std::hex << r.device_memory_ << ", " << std::dec << r.memory_size_in_bytes_ << "]";
    }

    #if 0
    Example append a device memory record
        device_memory_footprint dmf;
        dmf.insert_record( {id, mem, size} );
    Example delete a device memory record
        assert( dmf.device_memory_indexed_record_.find(mem) != dmf.device_memory_indexed_record_.end() );
        record r = *(dmf.device_memory_indexed_record_[mem]);
        dmf.delete( r );
    #endif
    struct device_memory_footprint
    {
        typedef std::list<std::shared_ptr<record>>::iterator record_itor;

        std::list<std::shared_ptr<record>>          record_;
        std::map<std::int64_t, record_itor>         tensor_id_indexed_record_;
        std::map<std::byte*,   record_itor>         device_memory_indexed_record_;
        std::multimap<std::int64_t, record_itor>    memory_size_in_bytes_indexed_record_;

        void insert_record( record const& r )
        {
            record_.emplace_front( std::make_shared<record>(r) );

            tensor_id_indexed_record_[r.tensor_id_] = record_.begin();
            device_memory_indexed_record_[r.device_memory_] = record_.begin();

            //memory_size_in_bytes_indexed_record_[r.memory_size_in_bytes_] = record_.begin();
            memory_size_in_bytes_indexed_record_.insert( std::make_pair(r.memory_size_in_bytes_, record_.begin() ) );
        }

        void update_record( record const& old_record, record const& new_record )
        {
            delete_record( old_record );
            insert_record( new_record );
        }

        void delete_record( record const& r )
        {
            if ( auto it = tensor_id_indexed_record_.find(r.tensor_id_); it != tensor_id_indexed_record_.end() )
            {
                record_itor record_itor_to_delete = it -> second;
                tensor_id_indexed_record_.erase( it );

                {
                //TODO:
                //    device_memory_indexed_record_.erase( r.device_memory_ );
                }
                {
                    auto rec = memory_size_in_bytes_indexed_record_.equal_range( r.memory_size_in_bytes_ );
                    for ( auto i = rec.first; i != rec.second; ++i )
                    {
                        if ( i->second == record_itor_to_delete )
                        {
                            memory_size_in_bytes_indexed_record_.erase( i );
                            break;
                        }
                    }
                }

                record_.erase( record_itor_to_delete );
            }
        }

    };//struct device_memory_footprint

}//namespace nnl

namespace std
{
    template<>
    struct hash<nnl::device_memory>
    {
        std::size_t operator()( nnl::device_memory const& dm ) const noexcept
        {
            return std::hash<std::byte*>{}(dm.address_) ^ std::hash<std::int64_t>{}(dm.size_in_bytes_);
        }
    }; //struct hash<nnl::device_memory>
}//namespace std

#endif//PBBMHSHQVGLAHLTKNTAUXYOGSLYGYOBKVYXJTDTCCHERVYPCHOUJEECFFCOWXDWTDDVPPUULL

