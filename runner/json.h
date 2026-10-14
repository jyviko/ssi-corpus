// Minimal JSON reader for the SSI evidence runner (own code, no dependencies).
// Supports objects, arrays, numbers, strings (no unicode escapes beyond \uXXXX
// pass-through), true/false/null. Throws std::runtime_error on malformed input.
#pragma once
#include <cctype>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mj {

struct Value {
  enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
  bool b = false;
  double n = 0.0;
  std::string s;
  std::vector<Value> a;
  std::map<std::string, Value> o;

  bool has(const std::string& k) const { return kind == Obj && o.count(k) > 0; }
  const Value& operator[](const std::string& k) const {
    auto it = o.find(k);
    if (kind != Obj || it == o.end()) throw std::runtime_error("missing key: " + k);
    return it->second;
  }
  const Value& operator[](size_t i) const {
    if (kind != Arr || i >= a.size()) throw std::runtime_error("bad array index");
    return a[i];
  }
  size_t size() const { return kind == Arr ? a.size() : 0; }
  double num() const {
    if (kind != Num) throw std::runtime_error("expected number");
    return n;
  }
  const std::string& str() const {
    if (kind != Str) throw std::runtime_error("expected string");
    return s;
  }
  bool boolean() const {
    if (kind != Bool) throw std::runtime_error("expected bool");
    return b;
  }
};

class Parser {
 public:
  explicit Parser(const std::string& t) : t_(t) {}
  Value parse() {
    Value v = value();
    ws();
    if (i_ != t_.size()) throw std::runtime_error("trailing characters");
    return v;
  }

 private:
  const std::string& t_;
  size_t i_ = 0;
  void ws() {
    while (i_ < t_.size() && std::isspace(static_cast<unsigned char>(t_[i_]))) ++i_;
  }
  char peek() {
    ws();
    if (i_ >= t_.size()) throw std::runtime_error("unexpected end");
    return t_[i_];
  }
  void expect(char c) {
    if (peek() != c) throw std::runtime_error(std::string("expected ") + c);
    ++i_;
  }
  Value value() {
    char c = peek();
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') {
      Value v;
      v.kind = Value::Str;
      v.s = string();
      return v;
    }
    if (t_.compare(i_, 4, "true") == 0) {
      i_ += 4;
      Value v;
      v.kind = Value::Bool;
      v.b = true;
      return v;
    }
    if (t_.compare(i_, 5, "false") == 0) {
      i_ += 5;
      Value v;
      v.kind = Value::Bool;
      return v;
    }
    if (t_.compare(i_, 4, "null") == 0) {
      i_ += 4;
      return Value();
    }
    return number();
  }
  Value object() {
    Value v;
    v.kind = Value::Obj;
    expect('{');
    if (peek() == '}') {
      ++i_;
      return v;
    }
    for (;;) {
      if (peek() != '"') throw std::runtime_error("expected key");
      std::string k = string();
      expect(':');
      v.o[k] = value();
      char c = peek();
      ++i_;
      if (c == '}') break;
      if (c != ',') throw std::runtime_error("expected , or }");
    }
    return v;
  }
  Value array() {
    Value v;
    v.kind = Value::Arr;
    expect('[');
    if (peek() == ']') {
      ++i_;
      return v;
    }
    for (;;) {
      v.a.push_back(value());
      char c = peek();
      ++i_;
      if (c == ']') break;
      if (c != ',') throw std::runtime_error("expected , or ]");
    }
    return v;
  }
  std::string string() {
    expect('"');
    std::string out;
    while (i_ < t_.size() && t_[i_] != '"') {
      char c = t_[i_++];
      if (c == '\\' && i_ < t_.size()) {
        char e = t_[i_++];
        switch (e) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'u': out += "\\u"; break;
          default: out += e;
        }
      } else {
        out += c;
      }
    }
    if (i_ >= t_.size()) throw std::runtime_error("unterminated string");
    ++i_;
    return out;
  }
  Value number() {
    ws();
    const char* begin = t_.c_str() + i_;
    char* end = nullptr;
    double d = std::strtod(begin, &end);
    if (end == begin) throw std::runtime_error("bad number");
    i_ += static_cast<size_t>(end - begin);
    Value v;
    v.kind = Value::Num;
    v.n = d;
    return v;
  }
};

inline Value parse(const std::string& text) { return Parser(text).parse(); }

}  // namespace mj
