#ifndef RDJQBNKCMTNWNPTQBBAOQPYWHGWHEEGOTSEYRSUXNCLDHNOMEQFAIKQCJPLCUQGOFKKJIFCNW
#define RDJQBNKCMTNWNPTQBBAOQPYWHGWHEEGOTSEYRSUXNCLDHNOMEQFAIKQCJPLCUQGOFKKJIFCNW
#include "../utility/utility.hpp"

namespace nnl
{
    struct stream
    {
    };

    struct memcpy_operation
    {
        std::byte*   src_memory_;
        std::byte*   dst_memory_;
        std::int64_t size_in_bytes_;

        void operator()(std::shared_ptr<stream> s) const
        {
            return (*this).impl( s );
        }

        virtual void impl(std::shared_ptr<stream> s) const = 0;
    };//memcpy_operation

    struct host2device_memory_operation : memcpy_operation
    {
        void impl(std::shared_ptr<stream> s) const override
        {
            //TODO
        }
    };//host2device_memory_operation

    struct device2host_memory_operation : memcpy_operation
    {
        void impl(std::shared_ptr<stream> s) const override
        {
            //TODO
        }
    };//device2host_memory_operation

    struct device2device_memory_operation : memcpy_operation
    {
        void impl(std::shared_ptr<stream> s) const override
        {
            //TODO
        }
    };//device2device_memory_operation


    struct arithmetic_operation
    {
        void impl(std::shared_ptr<stream> s) const override
        {
            return (*this).impl( s );
        }

        virtual void impl(std::shared_ptr<stream> s) const = 0;
    };//arithmetic_operation

    struct inference_node
    {
        // TODO: can here be a vector of operations?
        std::shared_ptr<arithmetic_operation> arithmetic_operation_;
        std::vector<std::shared_ptr<memcpy_operation>> memory_operations_;

        void operator()(std::vector<std::shared_ptr<stream>> sms) const
        {
            std::int64_t const n_mo = memory_operations_.size();
            for ( auto idx : range(n_mo) )
                (*(memory_operations_[idx]))( sms[idx] );
            (*arithmetic_operation_)( n_mo );
        }
    };//inference_node

    struct inference_sequence
    {
        std::list<inference_node> inference_nodes_;
        std::vector<std::shared_ptr<stream>> streams_;

        void append( std::shared_ptr<arithmetic_operation> ao, std::vector<std::shared_ptr<memcpy_operation>> mos )
        {
            inference_nodes_.emplace_back( ao, mos );
        }

        void operator()(std::shared_ptr<stream> sm, std::shared_ptr<stream> sa) const
        {
            for ( auto const& nd : inference_nodes_ )
            {
                std::int64_t n_streams = 1 + nd.memory_operations_.size();
                reserve_streams( n_streams );
                nd( streams_ );
                for ( auto idx : range( n_streams ) )
                    stream_sync( streams_[idx] );
            }
        }

        //https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
        void stream_sync(std::shared_ptr<stream> s) const //<-
        {
            //cudaStreamSynchronize
            //TODO
        }

        void reserve_streams( std::int64_t n )
        {
            if ( streams_.size() < n )
            {
                // append new streams to streams_
                //TODO
            }
        }
    };

}//namespace nnl
#endif//RDJQBNKCMTNWNPTQBBAOQPYWHGWHEEGOTSEYRSUXNCLDHNOMEQFAIKQCJPLCUQGOFKKJIFCNW
