#ifndef YVUPUJKKNAYJGGWTMKPEGNFAWCIIVBKKFPVVWWWGMXBKIOLMPHBPSJKNQFTXLMMHNBIUWIHAU
#define YVUPUJKKNAYJGGWTMKPEGNFAWCIIVBKKFPVVWWWGMXBKIOLMPHBPSJKNQFTXLMMHNBIUWIHAU
namespace nnl
{
    namespace
    {
        struct id
        {
            int value_;
            constexpr id( int value = 0 ) noexcept: value_{value} {}
        };

        inline int generate_uid() noexcept
        {
            static id id_generator;
            int ans = id_generator.value_;
            ++id_generator.value_;
            return ans;
        }
    };//namespace ceras_private

    template< typename Base >
    struct enable_id
    {
        int id_;
        enable_id() noexcept : id_ { generate_uid() } {}

        int id() const noexcept { return id_; }
    }; // struct enable_id

}//namespace nnl
#endif//YVUPUJKKNAYJGGWTMKPEGNFAWCIIVBKKFPVVWWWGMXBKIOLMPHBPSJKNQFTXLMMHNBIUWIHAU

