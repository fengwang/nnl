#ifndef UGJUIMKGCKYRPCBJPGQCFQBXAHNWYWEXWIVYIBHCDKIVLKUATSCXTEFVJBYJEJYAHOISPQNPL
#define UGJUIMKGCKYRPCBJPGQCFQBXAHNWYWEXWIVYIBHCDKIVLKUATSCXTEFVJBYJEJYAHOISPQNPL

#include "./node.hpp"
#include "./tensor.hpp"

namespace nnl
{

#if 0
    Layers are supposed to be abstract wrappers of Nodes.

    possible usage:

    layer input = Input();
    layer l1 = Dense( weights ...  )( input );
    ...
    layer output = attention( weights... )( ln );
    model m = Model( input, output );

    tensor t;
    auto result = m( t );
#endif

    struct layer
    {
        layer( node const& n, std::vector<layer> const& ls ) : node_{ n }, input_layers_{ std::make_shared<std::vector<layer>>( ls ) } { }

        node node_;
        //std::vector<layer> input_layers_; // <-- problem with stacked/recursive layers here!
        std::shared_ptr<std::vector<layer>> input_layers_; // <-- problem with stacked/recursive layers here!

        node underlying_node() const { return node_; }
        node& underlying_node() { return node_; }

        std::vector<layer> input_layers() const
        {
            better_assert( input_layers_ );
            return *input_layers_;
        }
    };


    inline bool operator < ( layer const& l1, layer const& l2 )
    {
        return l1.underlying_node().id() < l2.underlying_node().id();
    }

    inline bool operator == ( layer const& l1, layer const& l2 )
    {
        return l1.underlying_node().id() == l2.underlying_node().id();
    }


    inline std::ostream& operator<<( std::ostream& os, layer const& l )
    {
        os << "Layer[ <" << l.underlying_node().name() << "> : (";
        for ( auto il : l.input_layers() )
            os << il.underlying_node().name() << ", ";
        os << ") ]";
        return os;
    }


    inline auto Input( std::string const& name = std::string{"input"} )
    {
        return layer{ make_node( "Input", name ), std::vector<layer>{} };
    };

    template< Engine E >
    inline auto Dense( tensor<E> weight, tensor<E> bias, std::string const& name=std::string{"dense"} )
    {
        return [=]( layer input_layer )
        {
            auto current_node = make_node( "Dense", name );
            current_node.set_weights( std::vector<tensor<E>>{ {weight, bias,} } );
            return layer{ current_node, std::vector<layer>{ {input_layer,} } };
        };
    }



    inline auto Gelu( std::string const& name=std::string{"gelu"} )
    {
        return [=]( layer input_layer )
        {
            auto current_node = make_node( "Gelu", name );
            return layer{ current_node, std::vector<layer>{ {input_layer,} } };
        };
    }

    inline auto Add( std::string const& name=std::string{"add"} )
    {
        return [=]( layer lhs_input_layer, layer rhs_input_layer )
        {
            auto current_node = make_node( "Add", name );
            return layer{ current_node, std::vector<layer>{ {lhs_input_layer, rhs_input_layer,} } };
        };
    }

    inline auto Softmax( std::string const& name=std::string{"softmax"} )
    {
        return [=]( layer input_layer )
        {
            auto current_node = make_node( "Softmax", name );
            return layer{ current_node, std::vector<layer>{ {input_layer,} } };
        };
    }

    inline auto LayerNorm( std::string const& name=std::string{"layernorm"} )
    {
        return [=]( layer input_layer )
        {
            auto current_node = make_node( "LayerNorm", name );
            return layer{ current_node, std::vector<layer>{ {input_layer,} } };
        };
    }

    // This layer is used in gpt2
    template< Engine E >
    inline auto LayerNorm( tensor<E> alpha, tensor<E> beta, std::string const& name=std::string{"layernorm"} )
    {
        return [=]( layer input_layer )
        {
            auto current_node = make_node( "LayerNorm", name );
            current_node.set_weights( std::vector<tensor<E>>{alpha, beta,} );
            return layer{ current_node, std::vector<layer>{ {input_layer,} } };
        };
    }


    template< Engine E >
    inline auto Gemm( tensor<E> weight, std::string const& name=std::string{"gemm"} )
    {
        return [=]( layer input_layer )
        {
            auto current_node = make_node( "Gemm", name );
            current_node.set_weights( std::vector<tensor<E>>{ {weight,} } );
            return layer{ current_node, std::vector<layer>{ {input_layer,} } };
        };
    }

    template< Engine E >
    inline auto Linear( tensor<E> weight, tensor<E> bias, std::string name=std::string{"linear"} )
    {
        return [=]( layer input_layer )
        {
            auto current_node = make_node( "Linear", name );
            current_node.set_weights( std::vector<tensor<E>>{ {weight, bias,} } );
            return layer{ current_node, std::vector<layer>{ {input_layer,} } };
        };
    }

    template< Engine E >
    inline auto ScaledOffset( tensor<E> alpha, tensor<E> beta, std::string name=std::string{"scaledoffset"} )
    {
        return [=]( layer input_layer )
        {
            auto current_node = make_node( "ScaledOffset", name );
            current_node.set_weights( std::vector<tensor<E>>{ {alpha, beta,} } );
            return layer{ current_node, std::vector<layer>{ {input_layer,} } };
        };
    }

    inline auto ScaledOffset( std::string const& name=std::string{"scaledoffset"} )
    {
        return [=]( layer input_layer, layer alpha, layer beta )
        {
            auto current_node = make_node( "ScaledOffset", name );
            return layer{ current_node, std::vector<layer>{ {input_layer, alpha, beta,} } };
        };
    }

    // QueryProduct for Attention layer
    [[deprecated("Use Attention layer directly. This layer serves debug purpose only.")]]
    inline auto QueryProduct( std::string const& name={} )
    {
        return [=]( layer q, layer k )
        {
            auto current_node = make_node( "QueryProduct", name );
            return layer{ current_node, std::vector<layer>{ {q, k,} } };
        };
    }

    inline auto Multiply( std::string const& name={} )
    {
        return [=]( layer q, layer k )
        {
            auto current_node = make_node( "Multiply", name );
            return layer{ current_node, std::vector<layer>{ {q, k,} } };
        };
    }

    // ScaledMask for Attention layer
    [[deprecated("Use Attention layer directly. This layer does not catch the scale correctly.")]]
    inline auto ScaledMask( std::string const& name={} )
    {
        return [=]( layer input_layer )
        {
            auto current_node = make_node( "ScaledMask", name );
            return layer{ current_node, std::vector<layer>{ {input_layer,} } };
        };
    }

    inline auto Attention( std::string const& name=std::string{"attention"} )
    {
        return [=]( layer q, layer k, layer v )
        {
            auto current_node = make_node( "Attention", name );
            return layer{ current_node, std::vector<layer>{ {q, k, v,} } };
        };
    }

    template< Engine E >
    inline auto SelfAttention( tensor<E> w_q, tensor<E> b_q,
                               tensor<E> w_k, tensor<E> b_k,
                               tensor<E> w_v, tensor<E> b_v,
                               tensor<E> w_p, tensor<E> b_p, // <- the projection
                               std::string const& name=std::string{"selfattention"} )
    {
        std::string const& q_name = name + std::string{"_linear_q"};
        std::string const& k_name = name + std::string{"_linear_k"};
        std::string const& v_name = name + std::string{"_linear_v"};
        std::string const& p_name = name + std::string{"_linear_proj"};
        std::string const& a_name = name + std::string{"_attention"};

        return [=]( layer input_layer )
        {
            return Linear( w_p, b_p, p_name )
            (
                Attention( a_name )
                (
                    Linear(w_q, b_q, q_name)( input_layer ),
                    Linear(w_k, b_k, k_name)( input_layer ),
                    Linear(w_v, b_v, v_name)( input_layer )
                )
            );
        };
    }

#if 0
    // just for debugging purposes
    template< Engine E >
    inline auto MultiHeadAttention( std::vector<tensor<E>> const& w_qs, std::vector<tensor<E>> const& b_qs,
                                    std::vector<tensor<E>> const& w_ks, std::vector<tensor<E>> const& b_ks,
                                    std::vector<tensor<E>> const& w_vs, std::vector<tensor<E>> const& b_vs,
                                    std::vector<tensor<E>> const& w_ps, tensor<E> const& b_p, // <- the projection
                                    [[maybe_unused]] std::string const& name=std::string{} )
    {
        std::size_t const n = w_qs.size(); // <- number of heads
        better_assert( n != 0 );
        better_assert( n == b_qs.size() );
        better_assert( n == w_ks.size() );
        better_assert( n == b_ks.size() );
        better_assert( n == w_vs.size() );
        better_assert( n == b_vs.size() );
        better_assert( n == w_ps.size() );

        // same as SelfAttention, exception no bias applied at the last step
        auto const& UnbiasedSelfAttention = []( tensor<E> w_q, tensor<E> b_q,
                                                tensor<E> w_k, tensor<E> b_k,
                                                tensor<E> w_v, tensor<E> b_v,
                                                tensor<E> w_p )
        {
            return [=]( layer input_layer )
            {
                return Gemm( w_p ) // <-- no bias is applied
                (
                    Attention()
                    (
                        Linear(w_q, b_q)( input_layer ),
                        Linear(w_k, b_k)( input_layer ),
                        Linear(w_v, b_v)( input_layer )
                    )
                );
            };
        };

        return [=, &UnbiasedSelfAttention]( layer input_layer )
        {
            // The 1st layer uses SelfAttention
            layer ans = SelfAttention( w_qs[0], b_qs[0], w_ks[0], b_ks[0], w_vs[0], b_vs[0], w_ps[0], b_p )( input_layer );

            // the rest layers uses  UnbiasedSelfAttention
            for ( auto idx : range( 1UL, n ) )
            {
                layer _ans = UnbiasedSelfAttention( w_qs[idx], b_qs[idx], w_ks[idx], b_ks[idx], w_vs[idx], b_vs[idx], w_ps[idx] )( input_layer );
                ans = Add()( ans, _ans );
            }

            return ans;
        };
    }
#endif


    template< Engine E >
    inline auto MultiHeadAttention( tensor<E> w_att, tensor<E> b_att,
                                    tensor<E> w_pro, tensor<E> b_pro,
                                    std::int64_t const number_of_heads,
                                    [[maybe_unused]] std::string const& name=std::string{"multiheadattention"} )
    {
        return [=]( layer input_layer )
        {
            auto current_node = make_node( "MultiHeadAttention", name );
            current_node.add_attribute( make_attribute_heads( number_of_heads ) ); // <-- add attribute before setting weights
            current_node.set_weights( std::vector<tensor<E>>{ {w_att, b_att, w_pro, b_pro} } );
            return layer{ current_node, std::vector<layer>{ {input_layer,} } };
        };
    }

    template< Engine E >
    inline auto VocabularyProjection( tensor<E> alpha, tensor<E> beta, tensor<E> wte,
                                      [[maybe_unused]] std::string const& name=std::string{"vocabularyprojection"} )
    {
        return [=]( layer input_layer )
        {
            auto current_node = make_node( "VocabularyProjection", name );
            current_node.set_weights( std::vector<tensor<E>>{ {alpha, beta, wte} } );
            return layer{ current_node, std::vector<layer>{ {input_layer,} } };
        };
    }

    // gpt-2 layer
    template< Engine E >
    inline auto PositionwiseFeedForward( tensor<E> project_up_weight, tensor<E> project_up_bias,
                                           tensor<E> project_back_weight, tensor<E> project_back_bias,
                                           [[maybe_unused]] std::string const& name = std::string{"positionwisefeedforward"} )
    {
        return [=]( layer input_layer )
        {
            return Linear( project_back_weight, project_back_bias, name+std::string{"_outer_linear"} )
            (
                Gelu( name+std::string{"_middle_gelu"} )
                (
                    Linear( project_up_weight, project_up_bias, name+std::string{"_inner_linear"} )
                    (
                        input_layer
                    )
                )
            );
        };
    }


    // gpt-2 layer
    #if 0
        {
            "attn": {
                "c_attn": {"b": [3*n_embd], "w": [n_embd, 3*n_embd]},
                "c_proj": {"b": [n_embd], "w": [n_embd, n_embd]},
            },
            "ln_1": {"b": [n_embd], "g": [n_embd]},
            "ln_2": {"b": [n_embd], "g": [n_embd]},
            "mlp": {
                "c_fc": {"b": [4*n_embd], "w": [n_embd, 4*n_embd]},
                "c_proj": {"b": [n_embd], "w": [4*n_embd, n_embd]},
            },
        },
    #endif
    template< Engine E >
    inline auto Transformer
    (
        [[maybe_unused]] tensor<E> mlp_c_fc_w, [[maybe_unused]] tensor<E> mlp_c_fc_b,
        [[maybe_unused]] tensor<E> mlp_c_proj_w, [[maybe_unused]] tensor<E> mlp_c_proj_b,
        [[maybe_unused]] tensor<E> attn_c_attn_w, [[maybe_unused]] tensor<E> attn_c_attn_b,
        [[maybe_unused]] tensor<E> attn_c_proj_w, [[maybe_unused]] tensor<E> attn_c_proj_b,
        [[maybe_unused]] tensor<E> ln_1_g, [[maybe_unused]] tensor<E> ln_1_b,
        [[maybe_unused]] tensor<E> ln_2_g, [[maybe_unused]] tensor<E> ln_2_b,
        std::int64_t n_head,
        [[maybe_unused]] std::string const& name = std::string{"transformer"}
    )
    {
        return [=]( layer input_layer )
        {
            layer x = Add( name+std::string{"_add_1"})
                      (
                          input_layer,
                          // TODO: optimize weights to single merged  version
                          MultiHeadAttention( attn_c_attn_w, attn_c_attn_b, attn_c_proj_w, attn_c_proj_b, n_head, name+std::string{"_multiheadattention"} )
                          (
                            // TODO: optimize weights to single merged  version
                            LayerNorm( ln_1_g, ln_1_b, name+std::string{"_layernorm_1"} )( input_layer )
                          )
                      );
            return   Add( name+std::string{"_add_2"} )
                     (
                         x,
                         // TODO: optimize weights to single merged  version
                         PositionwiseFeedForward( mlp_c_fc_w, mlp_c_fc_b, mlp_c_proj_w, mlp_c_proj_b, name+std::string{"_positionwisefeedforward"} )
                         (
                            // TODO: optimize weights to single merged  version
                            LayerNorm( ln_2_g, ln_2_b, name+std::string{"_layernorm_2"} )( x )
                         )
                     );
        };
    }





}//namespace nnl

#endif//UGJUIMKGCKYRPCBJPGQCFQBXAHNWYWEXWIVYIBHCDKIVLKUATSCXTEFVJBYJEJYAHOISPQNPL

