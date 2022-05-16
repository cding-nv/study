// https://github.com/wangruofeng/Github_Blog/blob/master/%E5%B7%A5%E5%8E%82%E6%A8%A1%E5%BC%8F%E8%AF%A6%E8%A7%A3.md

#include <iostream>

using namespace std;

//��Ʒ�ȼ��ṹ--�ֻ�
class Phone {
  public:
    virtual ~Phone() {}; //��ɾ����ʱ���ֹ�ڴ�й¶
    virtual void call(string number) = 0;
};

class AndroidPhone : public Phone {
  public:
    void call(string number) {
        cout<<"AndroidPhone is calling..."<<endl;
    }
};

class IosPhone : public Phone {
  public:
    void call(string number) {
        cout<<"IosPhone is calling..."<<endl;
    }
};

//��Ʒ�ȼ��ṹ--ƽ��
class Pad {
  public:
    virtual ~Pad() {};
    virtual void playMovie() = 0;
};

class AndroidPad : public Pad {
  public:
    virtual void playMovie() {
        cout<<"AndriodPad is playing movie..."<<endl;
    }
};

class IosPad : public Pad {
  public:
    virtual void playMovie() {
        cout<<"IosPad is playing movie..."<<endl;
    }
};

class MobileFactory {
  public:
    virtual ~MobileFactory() {};
    virtual Phone* createPhone() = 0;
    virtual Pad* createPad() = 0;
};

class AndroidFactory : public MobileFactory {
  public:
    Phone* createPhone() {
        return new AndroidPhone();
    }
    Pad* createPad() {
        return new AndroidPad();
    }
};

class IosFactory : public MobileFactory {
  public:
    Phone* createPhone() {
        return new IosPhone();
    }

    Pad* createPad() {
        return new IosPad();
    }
};

int main() {
    MobileFactory*  androidCreator = new AndroidFactory();
    MobileFactory*  iosCreator = new IosFactory();
    Phone*  myAndroidPhone = androidCreator->createPhone();
    Pad* myAndroidPad = androidCreator->createPad();
    Phone* myIosPhone = iosCreator->createPhone();
    Pad* myIosPad = iosCreator->createPad();

    myAndroidPhone->call("123");
    myAndroidPad->playMovie();

    myIosPhone->call("456");
    myIosPad->playMovie();
    //����û�����ͷź��жϣ����Լ��жϺ��ͷ�

    return 0;
}