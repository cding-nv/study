#include <iostream>
#include <algorithm>
#include <string>

static void leftTrim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

static void rightTrim(std::string& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

int main() {
    std::string s = "   hello world !       ";
    std::cout << s ; std::cout << "End" <<std::endl;
    leftTrim(s);
    std::cout << s ; std::cout << "End" <<std::endl;
    rightTrim(s);
    std::cout << s ; std::cout << "End" <<std::endl;
    return 0;
}
