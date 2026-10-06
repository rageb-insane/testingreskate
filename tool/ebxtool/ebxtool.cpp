// ebxtool: small CLI over ReSkate's EBX reader/writer (Engine/Resource), used by
// the deck mod builder for the parts that need real EBX parsing.
//
//   ebxtool dump <file.ebx>
#include "Engine/Resource/ebx_document.h"
#include "Engine/Resource/ebx_writer.h"
#include "Engine/Resource/ebx_merge.h"
#include "bc7.h"
#define BCDEC_IMPLEMENTATION
#pragma warning(push, 0)
#include "External/bcdec/bcdec.h"
#pragma warning(pop)

#include <cstdio>
#include <functional>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace dingosdk::frostbite;
using namespace dingosdk::frostbite::ebx;

static std::vector<std::byte> read_all(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return {reinterpret_cast<std::byte*>(raw.data()), reinterpret_cast<std::byte*>(raw.data() + raw.size())};
}

static std::string hex(const std::byte* data, std::size_t size) {
    static const char* digits = "0123456789abcdef";
    std::string out;
    for (std::size_t i = 0; i < size; ++i) {
        auto v = std::to_integer<unsigned>(data[i]);
        out.push_back(digits[v >> 4]);
        out.push_back(digits[v & 15]);
    }
    return out;
}

static std::string type_name(const Document& doc, std::int32_t descriptor) {
    if (descriptor < 0 || static_cast<std::size_t>(descriptor) >= doc.types.size()) return "?";
    return doc.types[static_cast<std::size_t>(descriptor)].name;
}

static void dump_value(const Document& doc, const Value& value, int depth, std::ostream& out);

static void dump_object(const Document& doc, const Object& object, int depth, std::ostream& out) {
    out << type_name(doc, object.descriptor) << " {\n";
    for (const auto& field : object.fields) {
        out << std::string(static_cast<std::size_t>(depth + 1) * 2, ' ') << field.name << " = ";
        dump_value(doc, field.value, depth + 1, out);
        out << "\n";
    }
    out << std::string(static_cast<std::size_t>(depth) * 2, ' ') << "}";
}

static void dump_value(const Document& doc, const Value& value, int depth, std::ostream& out) {
    std::visit([&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) out << "<none>";
        else if constexpr (std::is_same_v<T, bool>) out << (v ? "true" : "false");
        else if constexpr (std::is_same_v<T, std::int64_t>) out << v;
        else if constexpr (std::is_same_v<T, std::uint64_t>) out << v << " (0x" << std::hex << v << std::dec << ")";
        else if constexpr (std::is_same_v<T, double>) out << v;
        else if constexpr (std::is_same_v<T, std::string>) out << '"' << v << '"';
        else if constexpr (std::is_same_v<T, Guid>) out << "guid(" << v.string() << ")";
        else if constexpr (std::is_same_v<T, Sha1>) out << "sha1(" << hex(v.bytes.data(), v.bytes.size()) << ")";
        else if constexpr (std::is_same_v<T, ResourceReference>) out << "res(0x" << std::hex << v.id << std::dec << ")";
        else if constexpr (std::is_same_v<T, PointerReference>) {
            if (v.kind == PointerKind::null) out << "null";
            else if (v.kind == PointerKind::internal) out << "ptr(#" << v.index << " " <<
                (v.index >= 0 && static_cast<std::size_t>(v.index) < doc.instances.size()
                     ? doc.instances[static_cast<std::size_t>(v.index)].instanceGuid.string() : "?") << ")";
            else {
                out << "import(#" << v.index;
                if (v.index >= 0 && static_cast<std::size_t>(v.index) < doc.imports.size()) {
                    const auto& imp = doc.imports[static_cast<std::size_t>(v.index)];
                    out << " file=" << imp.fileGuid.string() << " inst=" << imp.classGuid.string();
                }
                out << ")";
            }
        }
        else if constexpr (std::is_same_v<T, TypeReference>) out << "typeref(" << (v.primitive ? "prim" : type_name(doc, v.descriptor)) << ")";
        else if constexpr (std::is_same_v<T, BoxedReference>) {
            out << "boxed(" << v.encodedType;
            for (const auto& record : doc.boxedValues) {
                if (static_cast<std::int64_t>(record.offset) != v.dataOffset) continue;
                out << " at=" << (doc.dataStart + record.offset) << " bytes=" << hex(record.rawBytes.data(), record.rawBytes.size());
                break;
            }
            out << ")";
        }
        else if constexpr (std::is_same_v<T, std::shared_ptr<Object>>) { if (v) dump_object(doc, *v, depth, out); else out << "<null obj>"; }
        else if constexpr (std::is_same_v<T, Value::Array>) {
            out << "[" << v.size() << "]";
            for (std::size_t i = 0; i < v.size(); ++i) {
                out << "\n" << std::string(static_cast<std::size_t>(depth + 1) * 2, ' ') << "[" << i << "] ";
                dump_value(doc, v[i], depth + 1, out);
            }
        }
    }, value.data);
}

static int dump(const std::string& path) {
    const auto bytes = read_all(path);
    const auto doc = read_document(bytes);
    std::cout << "file " << doc.fileGuid.string() << " root " << doc.rootType << " serializedLayout=" << doc.serializedLayout
              << " hashedNames=" << doc.hashedReflectionNames << "\n";
    for (std::size_t i = 0; i < doc.imports.size(); ++i)
        std::cout << "import #" << i << " file=" << doc.imports[i].fileGuid.string() << " inst=" << doc.imports[i].classGuid.string() << "\n";
    for (std::size_t i = 0; i < doc.instances.size(); ++i) {
        const auto& inst = doc.instances[i];
        std::cout << "instance #" << i << " guid=" << inst.instanceGuid.string() << " exported=" << inst.exported
                  << " type=" << type_name(doc, inst.descriptor) << "\n  ";
        if (inst.object) dump_object(doc, *inst.object, 1, std::cout);
        std::cout << "\n";
    }
    return 0;
}

static void write_all(const std::string& path, const std::vector<std::byte>& bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

static int nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    throw std::runtime_error("bad hex digit");
}

static std::vector<std::byte> from_hex(const std::string& text) {
    std::string digits;
    for (char c : text) if (c != '-') digits.push_back(c);
    if (digits.size() % 2) throw std::runtime_error("odd hex length");
    std::vector<std::byte> out;
    for (std::size_t i = 0; i < digits.size(); i += 2)
        out.push_back(static_cast<std::byte>(nibble(digits[i]) * 16 + nibble(digits[i + 1])));
    return out;
}

// Inverse of Guid::string(): canonical text, first three fields byte-swapped.
static Guid parse_guid(const std::string& text) {
    const auto c = from_hex(text);
    if (c.size() != 16) throw std::runtime_error("bad guid " + text);
    Guid g;
    const int order[16] = {3, 2, 1, 0, 5, 4, 7, 6, 8, 9, 10, 11, 12, 13, 14, 15};
    for (int i = 0; i < 16; ++i) g.bytes[static_cast<std::size_t>(i)] = c[static_cast<std::size_t>(order[i])];
    return g;
}

// Walks "Field.Sub[2].Leaf" from an object, returning the value slot.
static Value& resolve(Document& doc, Object& object, const std::string& path) {
    std::size_t at = 0;
    Object* current = &object;
    Value* slot = nullptr;
    while (at < path.size()) {
        auto end = path.find_first_of(".[", at);
        const auto name = path.substr(at, end == std::string::npos ? std::string::npos : end - at);
        slot = nullptr;
        for (auto& field : current->fields) if (field.name == name) { slot = &field.value; break; }
        if (!slot) throw std::runtime_error("no field " + name + " in " + path);
        at = end == std::string::npos ? path.size() : end;
        while (at < path.size() && path[at] == '[') {
            const auto close = path.find(']', at);
            const auto index = std::stoul(path.substr(at + 1, close - at - 1));
            auto* array = std::get_if<Value::Array>(&slot->data);
            if (!array || index >= array->size()) throw std::runtime_error("bad index in " + path);
            slot = &(*array)[index];
            at = close + 1;
        }
        if (at < path.size() && path[at] == '.') {
            ++at;
            if (auto* inner = std::get_if<std::shared_ptr<Object>>(&slot->data); inner && *inner) {
                current = inner->get();
            } else if (auto* pointer = std::get_if<PointerReference>(&slot->data);
                       pointer && pointer->kind == PointerKind::internal) {
                current = doc.instances.at(static_cast<std::size_t>(pointer->index)).object.get();
            } else {
                throw std::runtime_error("cannot descend into " + path);
            }
        }
    }
    if (!slot) throw std::runtime_error("empty path");
    return *slot;
}

static Value parse_value(const std::string& typed) {
    const auto colon = typed.find(':');
    if (colon == std::string::npos) throw std::runtime_error("value needs a type prefix: " + typed);
    const auto type = typed.substr(0, colon), text = typed.substr(colon + 1);
    Value value;
    if (type == "str") value.data = text;
    else if (type == "u64") value.data = static_cast<std::uint64_t>(std::stoull(text, nullptr, 0));
    else if (type == "i64") value.data = static_cast<std::int64_t>(std::stoll(text, nullptr, 0));
    else if (type == "f64") value.data = std::stod(text);   // float fields too: the writer narrows to the field's type
    else if (type == "res") value.data = ResourceReference{std::stoull(text, nullptr, 0)};
    else if (type == "guid") value.data = parse_guid(text);
    else if (type == "sha1") {
        const auto bytes = from_hex(text);
        if (bytes.size() != 20) throw std::runtime_error("bad sha1");
        Sha1 sha;
        std::copy(bytes.begin(), bytes.end(), sha.bytes.begin());
        value.data = sha;
    } else throw std::runtime_error("unknown value type " + type);
    return value;
}

// ebxtool edit <in> <out> [ops...]
//   --file-guid G | --inst-guid I G | --import I FILE INST | --set I:Field.Path type:value
//   --set-import I:Field.Path FILE INST
static int edit(int argc, char** argv) {
    auto doc = read_document(read_all(argv[2]));
    for (int i = 4; i < argc; ++i) {
        const std::string op = argv[i];
        auto need = [&](int n) { if (i + n >= argc) throw std::runtime_error(op + " needs arguments"); };
        if (op == "--file-guid") { need(1); doc.fileGuid = parse_guid(argv[++i]); }
        else if (op == "--inst-guid") {
            need(2);
            const auto index = std::stoul(argv[++i]);
            doc.instances.at(index).instanceGuid = parse_guid(argv[++i]);
        } else if (op == "--import") {
            need(3);
            const auto index = std::stoul(argv[++i]);
            auto& imp = doc.imports.at(index);
            imp.fileGuid = parse_guid(argv[++i]);
            imp.classGuid = parse_guid(argv[++i]);
        } else if (op == "--set") {
            need(2);
            const std::string target = argv[++i];
            const auto colon = target.find(':');
            const auto index = std::stoul(target.substr(0, colon));
            auto& object = *doc.instances.at(index).object;
            auto& slot = resolve(doc, object, target.substr(colon + 1));
            auto replacement = parse_value(argv[++i]);
            // Keep the slot's numeric kind: the writer narrows by the field's declared type.
            if (std::holds_alternative<std::int64_t>(slot.data) && std::holds_alternative<std::uint64_t>(replacement.data))
                replacement.data = static_cast<std::int64_t>(std::get<std::uint64_t>(replacement.data));
            if (std::holds_alternative<std::uint64_t>(slot.data) && std::holds_alternative<std::int64_t>(replacement.data))
                replacement.data = static_cast<std::uint64_t>(std::get<std::int64_t>(replacement.data));
            if (slot.data.index() != replacement.data.index())
                throw std::runtime_error("type mismatch setting " + target);
            slot = std::move(replacement);
        } else if (op == "--append") {
            // --append I:ArrayPath: a copy of the array's last element goes on the end (set it with --set)
            need(1);
            const std::string target = argv[++i];
            const auto colon = target.find(':');
            auto& object = *doc.instances.at(std::stoul(target.substr(0, colon))).object;
            auto& slot = resolve(doc, object, target.substr(colon + 1));
            auto* array = std::get_if<Value::Array>(&slot.data);
            if (!array || array->empty()) throw std::runtime_error(target + " is not a non-empty array");
            std::function<Value(const Value&)> deep = [&](const Value& v) -> Value {
                Value out = v;
                if (auto* inner = std::get_if<std::shared_ptr<Object>>(&v.data); inner && *inner) {
                    auto copy = std::make_shared<Object>(**inner);
                    for (auto& field : copy->fields) field.value = deep(field.value);
                    out.data = copy;
                } else if (auto* list = std::get_if<Value::Array>(&v.data)) {
                    Value::Array items;
                    for (const auto& item : *list) items.push_back(deep(item));
                    out.data = std::move(items);
                }
                return out;
            };
            array->push_back(deep(array->back()));
        } else if (op == "--set-import") {
            // --set-import I:Field.Path FILE INST: point a pointer field at an import (added if new)
            need(3);
            const std::string target = argv[++i];
            const auto colon = target.find(':');
            auto& object = *doc.instances.at(std::stoul(target.substr(0, colon))).object;
            auto& slot = resolve(doc, object, target.substr(colon + 1));
            if (!std::holds_alternative<PointerReference>(slot.data))
                throw std::runtime_error(target + " is not a pointer field");
            const auto file = parse_guid(argv[++i]), inst = parse_guid(argv[++i]);
            std::int32_t index = -1;
            for (std::size_t at = 0; at < doc.imports.size(); ++at)
                if (doc.imports[at].fileGuid == file && doc.imports[at].classGuid == inst) index = static_cast<std::int32_t>(at);
            if (index < 0) {
                doc.imports.push_back({file, inst});
                index = static_cast<std::int32_t>(doc.imports.size() - 1);
            }
            slot.data = PointerReference{PointerKind::external, index};
        } else throw std::runtime_error("unknown edit op " + op);
    }
    write_all(argv[3], write_document(doc));
    return 0;
}

// ebxtool additem <collection> <out> <fileGuid> <instanceGuid> [arrayField=Items]
static int add_item(int argc, char** argv) {
    auto doc = read_document(read_all(argv[2]));
    const auto file = parse_guid(argv[4]), inst = parse_guid(argv[5]);
    const std::string field = argc > 6 ? argv[6] : "Items";
    auto* root = doc.instances.empty() ? nullptr : doc.instances.front().object.get();
    if (!root) throw std::runtime_error("collection has no root object");
    auto& slot = resolve(doc, *root, field);
    auto* array = std::get_if<Value::Array>(&slot.data);
    if (!array) throw std::runtime_error(field + " is not an array");
    std::int32_t index = -1;
    for (std::size_t at = 0; at < doc.imports.size(); ++at)
        if (doc.imports[at].fileGuid == file && doc.imports[at].classGuid == inst) index = static_cast<std::int32_t>(at);
    if (index < 0) {
        doc.imports.push_back({file, inst});
        index = static_cast<std::int32_t>(doc.imports.size() - 1);
    }
    for (const auto& element : *array)
        if (const auto* p = std::get_if<PointerReference>(&element.data); p && p->kind == PointerKind::external && p->index == index)
            throw std::runtime_error("item already listed");
    Value entry;
    entry.data = PointerReference{PointerKind::external, index};
    array->push_back(std::move(entry));
    write_all(argv[3], write_document(doc));
    return 0;
}

// ebxtool bc7enc <in.rgba> <width> <height> <out.bin>   (one mip level)
static int bc7_encode(char** argv) {
    const auto rgba = read_all(argv[2]);
    const int width = std::stoi(argv[3]), height = std::stoi(argv[4]);
    if (rgba.size() != static_cast<std::size_t>(width) * height * 4) throw std::runtime_error("RGBA size mismatch");
    const auto blocks = bc7::encode(reinterpret_cast<const std::uint8_t*>(rgba.data()), width, height);
    write_all(argv[5], {reinterpret_cast<const std::byte*>(blocks.data()), reinterpret_cast<const std::byte*>(blocks.data() + blocks.size())});
    return 0;
}

// ebxtool bc7dec <in.bin> <width> <height> <out.rgba>   (first mip level of the data)
static int bc7_decode(char** argv) {
    const auto data = read_all(argv[2]);
    const int width = std::stoi(argv[3]), height = std::stoi(argv[4]);
    const int bw = (width + 3) / 4, bh = (height + 3) / 4;
    if (data.size() < static_cast<std::size_t>(bw) * bh * 16) throw std::runtime_error("BC7 data is truncated");
    std::vector<std::byte> out(static_cast<std::size_t>(bw) * 4 * bh * 4 * 4);
    for (int by = 0; by < bh; ++by)
        for (int bx = 0; bx < bw; ++bx)
            bcdec_bc7(data.data() + (static_cast<std::size_t>(by) * bw + bx) * 16,
                      out.data() + (static_cast<std::size_t>(by) * 4 * bw * 4 + static_cast<std::size_t>(bx) * 4) * 4, bw * 4 * 4);
    std::vector<std::byte> cropped(static_cast<std::size_t>(width) * height * 4);
    for (int y = 0; y < height; ++y)
        std::memcpy(cropped.data() + static_cast<std::size_t>(y) * width * 4, out.data() + static_cast<std::size_t>(y) * bw * 4 * 4,
                    static_cast<std::size_t>(width) * 4);
    write_all(argv[5], cropped);
    return 0;
}

int main(int argc, char** argv) {
    try {
        const std::string command = argc > 1 ? argv[1] : "";
        if (command == "bc7enc" && argc >= 6) return bc7_encode(argv);
        if (command == "bc7dec" && argc >= 6) return bc7_decode(argv);
        if (command == "dump" && argc >= 3) return dump(argv[2]);
        if (command == "roundtrip" && argc >= 4) { write_all(argv[3], write_document(read_document(read_all(argv[2])))); return 0; }
        if (command == "edit" && argc >= 4) return edit(argc, argv);
        if (command == "additem" && argc >= 6) return add_item(argc, argv);
        if (command == "merge" && argc >= 5) {
            // ebxtool merge <base> <out> <edit>...: ReSkate's own merge of several mods' edits.
            const auto base = read_document(read_all(argv[2]));
            std::vector<Document> edits;
            for (int i = 4; i < argc; ++i) edits.push_back(read_document(read_all(argv[i])));
            std::vector<const Document*> pointers;
            for (const auto& edit : edits) pointers.push_back(&edit);
            MergeSummary summary;
            const auto merged = merge_documents(base, pointers, &summary);
            write_all(argv[3], write_document(merged));
            std::cout << "instances " << summary.instances << " arrayEntries " << summary.arrayEntries
                      << " renumbered " << summary.renumbered << "\n";
            return 0;
        }
        std::cerr << "usage: ebxtool dump <file> | roundtrip <in> <out> | edit <in> <out> ops... | additem <in> <out> <file> <inst> [field]\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
