#include <boost/algorithm/algorithm.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/multi_array.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <cmath>
#include <functional>
#include <type_traits>
#include <vector>
#include <memory>
namespace axlina
{
    using namespace boost::numeric;
    using namespace boost::math;
    using namespace boost::algorithm;

    template <typename T>
    concept scalar = std::is_floating_point_v<T>;

    using default_sclar_type = long double;

    namespace activator
    {
        template <scalar T = default_sclar_type>
        auto empty = [](const T& s) -> T { static_assert(false); };

        template <scalar T = default_sclar_type>
        auto linear = [](const T& s) -> T { return s; };

        template <scalar T = default_sclar_type>
        auto tanh = [](const T& s) -> T { return tanhl(s); };

        template <scalar T = default_sclar_type>
        auto sigmod = tanh<T>;

        template <scalar T = default_sclar_type>
        constexpr auto elu = [](const T& s) -> T {
            return s >= 0 ? s : elu<T>(powm1(constants::e<T>(), s));
        };

        template <scalar T = default_sclar_type>
        constexpr auto softp = [](const T& s) -> T {
            return log1p(powm1(constants::e<T>(), s) + 1);
        };

        template <scalar T = default_sclar_type>
        constexpr auto bentid =
            [](const T& s) -> T { return sqrtm1(s * s + 1) / 2 + s; };
    };

    template <scalar T = default_sclar_type>
    class node
    {
    public:
        using vector_type    = ublas::vector<T>;
        using activator_type = std::function<T(const T&)>;

    private:
        vector_type    weight;
        T              bias;
        activator_type transfer_func;

    public:
        node() = default;
        node(const vector_type& init_weight,
             activator_type     init_func = activator::sigmod<T>)
            : weight(init_weight)
            , transfer_func(init_func){};
        ~node() = default;
        auto calculate(const vector_type& input)
        {
            T r = ublas::inner_prod(ublas::trans(weight), input);
            return transfer_func(r + bias);
        }
    };
    template <scalar T = default_sclar_type, typename... Nal>
    class layer
    {
    public:
        using node_type = node<T>;
        using list_type = std::vector<std::shared_ptr<node_type>>;

    private:
        list_type node_list;

    public:
        layer(std::size_t sz, Nal&&... arg_list)
        {
            for (size_t i = 0; i < sz; i++)
            {
                node_list.emplace_back(
                    std::make_shared<T, std::forward<Nal>(arg_list)>...);
            }
        };
        ~layer() = default;
        //constexpr auto begin(){return node_list.begin();};
        const list_type& list() const
        {
            return node_list;
        }
    };

    namespace linker
    {
        template <scalar T = default_sclar_type>
        using weak_list_type = std::vector<std::weak_ptr<node<T>>>;
        template <scalar T = default_sclar_type>
        using func_type = std::function<std::unique_ptr<weak_list_type<T>>(
            const layer<T>& lhs, const node<T>& rhs)>;
        template <scalar T = default_sclar_type>
        auto empty = [](const layer<T>& lhs, const node<T>& rhs)
            -> std::unique_ptr<weak_list_type<T>> { static_assert(false); };
        template <scalar T = default_sclar_type>
        auto full = [](const layer<T>& lhs,
                       const node<T>& rhs) -> std::unique_ptr<weak_list_type<T>>
        {
            auto rtn = std::make_unique<weak_list_type<T>>();
            for (auto& v : lhs.list())
            {
                rtn->push_back(std::weak_ptr(v));
            }
            return rtn;
        };
    }
    template <scalar T = default_sclar_type>
    class connection
    {
    public:
        using node_type        = node<T>;
        using clref_layer_type = const layer<T>&;
        using list_type        = std::vector<linker::weak_list_type<T>>;
        using linker_type      = linker::func_type<T>;

    private:
        clref_layer_type lv;
        clref_layer_type rv;
        linker_type      linker_func;
        list_type        ref_list;

    public:
        connection(clref_layer_type lhs, clref_layer_type rhs,
                   linker_type linker = linker::full<T>)
            : lv(lhs)
            , rv(rhs)
            , linker_func(linker)
            , ref_list(rhs.list().size())
        {
            for (const auto& v : rhs.list())
            {
                ref_list.emplace_back(*std::move(linker_func(lhs, *v)));
            }
        }
        constexpr auto get(std::size_t n)
        {
            return ref_list[n];
        }
    };
} // namespace axlina
int main()
{
    auto lx = axlina::layer<double>(16);
    auto ly = axlina::layer<double>(16);
    auto cn = axlina::connection<double>(lx, ly);
}
