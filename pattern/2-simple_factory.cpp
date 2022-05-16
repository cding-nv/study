//https://github.com/wangruofeng/Github_Blog/blob/master/%E5%B7%A5%E5%8E%82%E6%A8%A1%E5%BC%8F%E8%AF%A6%E8%A7%A3.md

#include <iostream>

using namespace std;

class Phone {
  public:
    virtual ~Phone() {}; //在删除的时候防止内存泄露
    virtual void call(string number) = 0;
};

class AndroidPhone : public Phone {
  public:
    void call(string number) {
        cout << "AndroidPhone is calling " << number <<endl;
    }
};

class IosPhone : public Phone {
  public:
    void call(string number) {
        cout << "IosPhone is calling..." << number << endl;
    }
};

class PhoneFactory {
  public:
    Phone* createPhone(string phoneName) {
        if(phoneName == "AndroidPhone") {
            return new AndroidPhone();
        } else if(phoneName == "iOSPhone") {
            return new IosPhone();
        }
        return NULL;
    }
};

int main() {
    PhoneFactory factory;
    Phone* myAndroid = factory.createPhone("AndroidPhone");
    Phone* myIPhone = factory.createPhone("iOSPhone");
    if(myAndroid) {
        myAndroid->call("123");
        delete myAndroid;
        myAndroid = NULL;
    }

    if(myIPhone) {
        myIPhone->call("456");
        delete  myIPhone;
        myIPhone = NULL;
    }
    return 0;
}