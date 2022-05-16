// https://github.com/wangruofeng/Github_Blog/blob/master/%E5%B7%A5%E5%8E%82%E6%A8%A1%E5%BC%8F%E8%AF%A6%E8%A7%A3.md

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
        cout << "AndroidPhone is calling... " << number << endl;
    }
};

class iOSPhone : public Phone {
  public:
    void call(string number) {
        cout<<"iOSPhone is calling... " << number <<endl;
    }
};

class PhoneFactory {
  public:
    virtual ~PhoneFactory() {};
    virtual Phone* createPhone() = 0;
};

class AndroidPhoneFactory : public PhoneFactory {
  public:
    virtual Phone* createPhone() {
        return new AndroidPhone();
    }
};

class iOSPhoneFactory : public PhoneFactory {
  public:
    virtual Phone* createPhone() {
        return new iOSPhone();
    }
};

int main() {
    PhoneFactory*  androidCreator = new AndroidPhoneFactory();
    PhoneFactory*  iosCreator = new iOSPhoneFactory();
    Phone*  myAndroid = androidCreator->createPhone();
    Phone* myIPhone = iosCreator->createPhone();
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

    delete androidCreator;
    delete iosCreator;
    return 0;
}