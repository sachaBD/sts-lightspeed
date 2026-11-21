#ifndef STS_LIGHTSPEED_STATEFIELDS_H
#define STS_LIGHTSPEED_STATEFIELDS_H

// A combat state as a walk over its leaves, for code that stores states or compares them: every field, in a
// fixed order, down to scalars (integers, enums, bools) and RNGs, which are left whole so a caller can store
// their position rather than their bits.
//
// One walk both reads and fills: on a const state the leaves are const; on a mutable one the callback may assign
// them. A container's size is a leaf visited before its elements, so a fill sizes the container from it.
// Each struct names its own fields beside their declarations (visitFields); this file only walks containers.

#include <array>
#include <bitset>
#include <cstdint>
#include <map>
#include <type_traits>
#include <vector>
#include <variant>
#include <stdexcept>

#include "data_structure/fixed_list.h"
#include "game/Random.h"

namespace sts {

    template <class T, class Leaf>
    void forEachField(T &value, Leaf &&leaf);

    namespace state_fields {
        template <class T> struct IsResizable : std::false_type {};
        template <class T> struct IsResizable<std::vector<T>> : std::true_type {};
        template <class T, int N> struct IsResizable<fixed_list<T, N>> : std::true_type {};

        template <class T> struct IsMap : std::false_type {};
        template <class K, class V> struct IsMap<std::map<K, V>> : std::true_type {};

        template <class T> struct IsBitset : std::false_type {};
        template <std::size_t N> struct IsBitset<std::bitset<N>> : std::true_type {};

        template <class T> struct IsVariant : std::false_type {};
        template <class... T> struct IsVariant<std::variant<T...>> : std::true_type {};

        template <std::size_t I = 0, class Variant>
        void selectVariant(Variant& value, std::size_t index) {
            if constexpr (I < std::variant_size_v<Variant>) {
                if (index == I) value.template emplace<I>();
                else selectVariant<I + 1>(value, index);
            } else {
                throw std::invalid_argument{"invalid state variant"};
            }
        }

        template <class T> struct IsArray : std::is_array<T> {};
        template <class T, std::size_t N> struct IsArray<std::array<T, N>> : std::true_type {};
    }

    template <class T, class Leaf>
    void forEachField(T &value, Leaf &&leaf) {
        using U = std::remove_const_t<T>;
        constexpr bool filling = !std::is_const_v<T>;
        auto walk = [&leaf](auto &field) { forEachField(field, leaf); };

        if constexpr (std::is_arithmetic_v<U> || std::is_enum_v<U> || std::is_same_v<U, Random>) {
            leaf(value);
        } else if constexpr (state_fields::IsVariant<U>::value) {
            std::uint64_t index = value.index();
            leaf(index);
            if constexpr (filling) state_fields::selectVariant(value, index);
            std::visit(walk, value);
        } else if constexpr (state_fields::IsBitset<U>::value) {
            std::uint64_t bits = value.to_ullong();
            leaf(bits);
            if constexpr (filling) value = U(bits);
        } else if constexpr (state_fields::IsArray<U>::value) {
            for (auto &element : value) walk(element);
        } else if constexpr (state_fields::IsResizable<U>::value) {
            std::uint64_t size = value.size();
            leaf(size);
            if constexpr (filling) value.resize(static_cast<int>(size));
            for (auto &element : value) walk(element);
        } else if constexpr (state_fields::IsMap<U>::value) {
            std::uint64_t size = value.size();
            leaf(size);
            if constexpr (filling) {
                value.clear();
                for (std::uint64_t i = 0; i < size; ++i) {
                    typename U::key_type key{};
                    typename U::mapped_type mapped{};
                    walk(key);
                    walk(mapped);
                    value.emplace(key, mapped);
                }
            } else {
                for (auto &[key, mapped] : value) {
                    walk(key);
                    walk(mapped);
                }
            }
        } else {
            U::visitFields(value, walk);  // a struct: it lists its own fields
        }
    }

}

#endif //STS_LIGHTSPEED_STATEFIELDS_H
