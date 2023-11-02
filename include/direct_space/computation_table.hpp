#ifndef GJNNCNPXHXWXBHBRFAXUOHOBTYHHCAULGUNCSHKTLPYASYRROOSORAETDSYYUINNLGXBOBANK
#define GJNNCNPXHXWXBHBRFAXUOHOBTYHHCAULGUNCSHKTLPYASYRROOSORAETDSYYUINNLGXBOBANK

#include "../utility/utility.hpp"

namespace nnl
{

    //
    // 1. The computation that invokes CUDA kernel
    // 2. Host to Device memory copy of the output tensors for the next computation kernel
    // 3. Device to Host memory copy to save the GPU memory
    // 4. Intermediate GPU results that will not be reused can be dismissed without copying back to Host
    // 5. Weight preloading for next computation block
    // 6. Dismiss weights that has been preloaded
    // 7. Pre-allocate output tensor memory, should be identical to 5.
    //
    // During the inference time,
    // 1. a node in field arithmetic_operation_ will invoke kernel calls
    // 2. a node in field host2device_operations_ will invoke host2device memcpy for the output tensors
    template< typename Node >
    struct computation_block
    {
        // 1
        Node arithmetic_operation_;
        // 2
        std::set<Node> host2device_operations_;
        // 3
        std::set<Node> device2host_operations_;
        // 4
        std::set<Node> device2host_dismiss_operations_;
        // 5
        Node weight_preloading_;
        // 6
        Node weight_dismissing_;
        // 5
        Node output_preallocating_;
    };

    template< typename Node >
    inline std::ostream& operator << ( std::ostream& os, computation_block<Node> const& cb )
    {
        if constexpr ( has_ostream_v<Node> )
        {
            os << "ComputationBlock:\n";
            os << "\tArithmeticOperation: " << cb.arithmetic_operation_ << "\n";
            os << "\tHost2DeviceOperations: [";
            for ( auto const& ho : cb.host2device_operations_ )
                os << ho << ", ";
            os << "]\n";
            os << "\tDevice2HostOperations: [";
            for ( auto const& dh : cb.device2host_operations_ )
                os << dh << ", ";
            os << "]\n";
            os << "\tDeviceDismissOperations: [";
            for ( auto const& ho : cb.device2host_dismiss_operations_ )
                os << ho << ", ";
            os << "]\n";
            os << "\tWeightPreload: " << cb.weight_preloading_ << "\n";
            os << "\tWeightDismiss: " << cb.weight_dismissing_ << "\n";
            os << "\tOutputPreallocate: " << cb.output_preallocating_ << "\n";
        }
        return os;
    }

    template< typename Node >
    auto make_computation_table( std::vector<Node> const& computation_order,
                                 std::map<Node, std::vector<Node>> const& edges )
    {
        std::vector<computation_block<Node>> ans;

        // initialize empty computation block as the first one
        ans.emplace_back( Node{nullptr}, std::set<Node>{}, std::set<Node>{}, std::set<Node>{}, Node{nullptr}, Node{nullptr} );

        // collect all arithmetic operations
        for ( auto const& nd : computation_order )
        {
            if ( nd.is_memcpy_only() )
            {
                if constexpr ( debug_mode )
                {
                    std::cout << "Collecting arithmetic operations: skip " << nd << " as it is memcpy only\n";
                }
                continue;
            }
            ans.emplace_back( nd, std::set<Node>{}, std::set<Node>{}, std::set<Node>{}, Node{nullptr}, Node{nullptr} );
        }

        // put inputs of arithemtic operations to the prev blocks_
        for ( auto idx : range( 1UL, ans.size() ) )
        {
            auto input_pos = edges.find(ans[idx].arithmetic_operation_);
            assert( input_pos != edges.end() );
            std::vector<Node> const& inputs = input_pos->second;
            ans[idx-1].host2device_operations_.insert( inputs.begin(), inputs.end() );
            ans[idx-1].weight_preloading_ = input_pos->first;
            ans[idx].weight_dismissing_ = input_pos->first;
        }

        if constexpr (debug_mode )
        {
            std::cout << "After putting arithmetic:\n";
            for ( auto const& cb : ans )
                std::cout << cb << std::endl;
            std::cout << std::endl;
        }

        //if ( ans.size() == 1 )
        if ( ans.size() <= 1 )
            return ans;

        {
            std::set<Node> current_in_device;
            for ( auto idx : range( ans.size() ) )
            {
                auto& [ao, h2ds, d2hs, dms, wl, wd, opl]  = ans[idx];
                auto& [ao_, h2ds_, d2hs_, dms_, wl_, wd_, opl_]  = ans[idx+1];

                {
                    current_in_device.insert( ao );
                    current_in_device.insert( h2ds.begin(), h2ds.end() );
                    std::set<Node> _current_in_device;
                    std::set_difference( current_in_device.begin(), current_in_device.end(), d2hs.begin(), d2hs.end(),
                                         std::inserter( _current_in_device, _current_in_device.end() ) );
                    std::swap( _current_in_device, current_in_device );

                }

                if ( idx == ans.size() - 1 )
                    continue;

                // remove consecutive host2device and device2host
                {
                    std::set<Node>& target_d2hs = d2hs_;
                    std::set<Node> _target_d2hs;
                    std::set_difference( current_in_device.begin(), current_in_device.end(), h2ds_.begin(), h2ds_.end(),
                            std::inserter( _target_d2hs, _target_d2hs.end() ) );
                    std::swap( _target_d2hs, target_d2hs );
                    {
                        if ( idx < ans.size() - 2 )
                        {
                            auto& [ao__, h2ds__, d2hs__, dms__, wl__, wd__, opl__] = ans[idx+2];
                            std::set<Node> hdn;
                            std::set_difference( d2hs_.begin(), d2hs_.end(), h2ds__.begin(), h2ds__.end(),
                                    std::inserter( hdn, hdn.end() ) );
                            std::swap( d2hs_, hdn );
                        }
                    }
                }

                {
                    std::set<Node>& target_h2ds = h2ds_;
                    std::set<Node> _target_h2ds;
                    std::set_difference( target_h2ds.begin(), target_h2ds.end(), current_in_device.begin(), current_in_device.end(),
                           std::inserter( _target_h2ds, _target_h2ds.end() ) );
                    std::swap( target_h2ds, _target_h2ds );
                }

            }//end for
        }

        if constexpr (debug_mode )
        {
            std::cout << "After adjusting h2d and d2h:\n";
            for ( auto const& cb : ans )
                std::cout << cb << std::endl;
            std::cout << std::endl;
        }

        //std::set<Node> host2device_dismiss_operations_;
        {
            // Not copying constant/input node back from device to host
            for ( auto& cb : ans )
            {
                auto& d2hs = cb.device2host_operations_;
                auto& dms = cb.device2host_dismiss_operations_;
                std::set<Node> _d2hs;
                dms.clear();
                for ( auto nd : d2hs )
                {
                    if ( ! nd ) //skip empty nodes
                    {
                        continue;
                    }

                    if ( nd.is_memcpy_only() )
                    {
                        dms.insert( nd );
                        if constexpr (debug_mode)
                        {
                            spdlog::debug( "node {} is memcpy only, moved to device2host_dismiss operations.", nd.name() );
                        }
                    }
                    else
                    {
                        _d2hs.insert( nd );
                        if constexpr (debug_mode)
                        {
                            spdlog::debug( "node {} is NOT memcpy only, moved to device2host operations.", nd.name() );
                        }
                    }
                }
                std::swap( _d2hs, d2hs );
            }
        }

        if constexpr (debug_mode )
        {
            std::cout << "After removing constant/input node Device2Host:\n";
            for ( auto const& cb : ans )
                std::cout << cb << std::endl;
            std::cout << std::endl;
        }

        {
            // Not copying intermediate node that is not to be used afterward from device to host
            std::set<Node> d2hs_records;
            for ( auto& cb : ans | std::views::reverse )
            {
                auto& [ao, h2ds, d2hs, dms, wl, wd, opl] = cb;
                std::set<Node> _d2hs;
                for ( auto const& nd : d2hs )
                {
                    if ( d2hs_records.find( nd ) == d2hs_records.end() && (!nd.is_output_node()) )
                    {
                        dms.insert( nd );
                    }
                    else
                    {
                        //d2hs_records.insert( nd );
                        _d2hs.insert( nd );
                    }
                    d2hs_records.insert( nd );
                }
                std::swap( _d2hs, d2hs );
            }
        }

        if constexpr (debug_mode )
        {
            std::cout << "After removing intermediate nodes that are not to be used afterwards:\n";
            for ( auto const& cb : ans )
                std::cout << cb << std::endl;
            std::cout << std::endl;
        }

        // skip host2device if arithemtic is in the same block
        for ( auto& cb : ans )
        {
            auto& [ao, h2ds, d2hs, dms, wl, wd, opl]  = cb;
            h2ds.erase( ao );
        }

        if constexpr (debug_mode )
        {
            std::cout << "After removing Host2Device operations for arithemtic operations in a same block:\n";
            for ( auto const& cb : ans )
                std::cout << cb << std::endl;
            std::cout << std::endl;
        }

        // output preallocation should be identical to weight_preloading_
        for ( auto& cb : ans )
        {
            auto& [ao, h2ds, d2hs, dms, wl, wd, opl]  = cb;
            opl = wl;
        }

        if constexpr (debug_mode )
        {
            std::cout << "After removing duplicated preallocation and weight preloading:\n";
            for ( auto const& cb : ans )
                std::cout << cb << std::endl;
            std::cout << std::endl;
        }



        return ans;
    }

    template< typename Node >
    auto make_computation_table( graph<Node>& g )
    {
        return make_computation_table( g.computation_order(), g.edges() );
    }

    template< typename Node >
    auto make_mega_computation_table( std::vector<computation_block<Node>> const& table )
    {
        // TODO: minimize host->device and device->host traffic by stacking arithemtic operations
        return 0;
    }

}//namespace nnl

#endif//GJNNCNPXHXWXBHBRFAXUOHOBTYHHCAULGUNCSHKTLPYASYRROOSORAETDSYYUINNLGXBOBANK

