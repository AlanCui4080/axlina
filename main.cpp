#include <boost/numeric/ublas/vector.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/algorithm/algorithm.hpp>
#include <boost/multi_array.hpp>
#include <functional>
#include <type_traits>
#include <vector>
#include <cmath>
namespace axlina {

    using namespace boost::numeric;
    using namespace boost::math;
    using namespace boost::algorithm;

    template <typename T>
    concept scalar = std::is_floating_point_v<T>;

    namespace activator {
        template <scalar T>
        auto linear = [](const T& s) -> T { return s; };
        template <scalar T>
        auto tanh = [](const T& s) -> T { return tanhl(s); };
        template <scalar T>
        auto sigmod = tanh<T>;
        template <scalar T>
        constexpr auto elu = [](const T& s) -> T {
            return s >= 0 ? s : elu<T>(powm1(constants::e<T>(), s));
        };
        template <scalar T>
        constexpr auto softp = [](const T& s) -> T {
            return log1p(powm1(constants::e<T>(), s) + 1);
        };
        template <scalar T>
        constexpr auto bentid =
            [](const T& s) -> T { return sqrtm1(s * s + 1) / 2 + s; };
    };

    template <scalar T>
    class node {
    public:
        using vector_type    = ublas::vector<T>;
        using scalar_type    = T;
        using activator_type = std::function<scalar_type(const scalar_type&)>;

    private:
        vector_type    weight;
        scalar_type    bias;
        activator_type transfer_func;

    public:
        node() = default;
        node(const vector_type& init_weight,
             activator_type     init_func = activator::sigmod<scalar_type>)
            : weight(init_weight)
            , transfer_func(init_func){};
        ~node() = default;
        auto calculate(const vector_type& input) {
            scalar_type r = ublas::inner_prod(ublas::trans(weight), input);
            return transfer_func(r + bias);
        }
    };
    template <scalar T>
    class layer {
    public:
        using node_type = node<T>;

    private:
        size_t                 node_size;
        std::vector<node_type> node_list;

    public:
        layer(std::size_t sz)
            : node_list(sz)
            , node_size(sz){};
        ~layer() = default;
    };
    template <scalar T>
    using linker_type =
        std::function<std::vector<const node<T>&> &&
                      (const layer<T>& lhs, const node<T>& rhs)>;
    namespace linker {
        template <scalar T>
        constexpr auto full =
            [](const std::vector<T>& lhs, const node<T>& rhs) {
                std::vector<const node<T>&> rtn;
                for (const auto& v : lhs) {
                    rtn.push_back(&rtn);
                }
                return rtn;
            };
    }
    template <scalar T>
    class connection {
    public:
        using node_type = node<T>;

    private:
        const layer<T>&                          lv;
        const layer<T>&                          rv;
        linker_type<T>                           linker_func;
        std::vector<std::vector<const node<T>&>> ref_list;

    public:
        connection(const std::vector<node<T>>& lhs,
                   const std::vector<node<T>>& rhs,
                   linker_type<T>              linker = linker::full<T>)
            : lv(lhs)
            , rv(rhs)
            , linker_func(linker)
            , ref_list(rhs.size()) {
            for (const auto& v : rhs) {
                v.emplace_back(linker_func(lhs, v));
            }
        }
    };
} // namespace axlina
int main()
{
    
}
