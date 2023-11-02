#include "./gpt2_tokenizer.hpp"

#include <string>

int main()
{
    std::string const vocab_file{"examples/gpt2-1558M/assets/vocab.json"};
    std::string const merges_file{"examples/gpt2-1558M/assets/merges.txt"};
    auto tk = make_gpt2_tokenizer( vocab_file, merges_file );
    //std::string str{ "Alan Turing theorized that computers would one day become" };
    std::string str{ "Not all heroes wear capes." };
    std::cout << "Testing with string : " << str << "\n";
    auto enc = tk->encode( str );
    {
        std::cout << "Encoding:\n";
        for ( auto _e : enc )
            std::cout << _e << " ";
        std::cout << "\n";
    }

    auto dec = tk->decode( enc );
    std::cout << "Decoded to :\n" << dec << "\n";

    return 0;
}



