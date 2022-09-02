//https://blog.csdn.net/xiaolewennofollow/article/details/52559364
#include <vector>  
#include <string>  
#include <iostream>  

//std::move 用于指示对象 t 可以“被移动”，即允许从 t 到另一对象的有效率的资源传递
//std::move 执行一个无条件的转化到右值。它本身并不移动任何东西；
//std::forward 把其参数转换为右值，仅仅在那个参数被绑定到一个右值时；
//std::move 和std::forward 在运行时（runtime）都不做任何事

//有了右值引用和转移语义，我们在设计和实现类时，对于需要动态申请大量资源的类，应该设计转移构造函数和转移赋值函数，以提高应用程序的效率

struct President  
{  
    std::string name;  
    std::string country;  
    int year;  

    President(std::string p_name, std::string p_country, int p_year)  
        : name(std::move(p_name)), country(std::move(p_country)), year(p_year)  
    {  
        std::cout << "I am being constructed.\n";  
    }
    President(const President& other)
        : name(std::move(other.name)), country(std::move(other.country)), year(other.year)
    {
        std::cout << "I am being copy constructed.\n";
    }
    President(President&& other)  
        : name(std::move(other.name)), country(std::move(other.country)), year(other.year)  
    {  
        std::cout << "I am being moved.\n";  
    }  
    President& operator=(const President& other);  
};  

int main()  
{  
    std::vector<President> elections;  
    std::cout << "emplace_back:\n";  
    elections.emplace_back("Nelson Mandela", "South Africa", 1994); //没有类的创建  

    std::vector<President> reElections;  
    std::cout << "\npush_back:\n";  
    reElections.push_back(President("Franklin Delano Roosevelt", "the USA", 1936));  

    std::cout << "\nContents:\n";  
    for (President const& president: elections) {  
       std::cout << president.name << " was elected president of "  
            << president.country << " in " << president.year << ".\n";  
    }  
    for (President const& president: reElections) {  
        std::cout << president.name << " was re-elected president of "  
            << president.country << " in " << president.year << ".\n";  
    }
    return 0;
}
