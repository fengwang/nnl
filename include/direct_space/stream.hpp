#ifndef QVKVEVYJVKAWNSYLIACKSCGUNFOMSMWYCSWLREOWTDMMYUDEMOBHPAQQTQXSDTTYPWKBSDHDN
#define QVKVEVYJVKAWNSYLIACKSCGUNFOMSMWYCSWLREOWTDMMYUDEMOBHPAQQTQXSDTTYPWKBSDHDN

#include "../utility/utility.hpp"
#include "./engine.hpp"

namespace nnl
{

    template< Engine E>
    struct stream;

    template<>
    struct stream<cuda_engine> : enable_id<stream<cuda_engine>>
    {
        struct cuda_stream;
        std::unique_ptr<cuda_stream> stream_;

        void synchronize();

        stream();
        ~stream();

        stream( stream&& );
        stream& operator = ( stream&& );

        // too expensive to use
        stream( stream const& ) = delete;
        stream& operator = ( stream const& ) = delete;
    };//struct stream<cuda_engine>

    inline std::ostream& operator << ( std::ostream& os, stream<cuda_engine> const& sm )
    {
        os << "Stream: " << sm.id() << "\n";
        return os;
    }

}//namespace nnl

#endif//QVKVEVYJVKAWNSYLIACKSCGUNFOMSMWYCSWLREOWTDMMYUDEMOBHPAQQTQXSDTTYPWKBSDHDN

