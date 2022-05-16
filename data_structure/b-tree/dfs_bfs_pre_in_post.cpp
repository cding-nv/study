// https://blog.csdn.net/mingwanganyu/article/details/72033122


#include <iostream>
#include <stdlib.h>
#include <malloc.h>
#include <stack>
#include <queue>
using namespace std;

typedef struct Node {
    char data;
    struct Node *lchild;
    struct Node *rchild;
} *Tree;
//Tree 是一个node指针的类型定义

int index = 0;  //全局索引变量
//二叉树构造器,按先序遍历顺序构造二叉树
//无左子树或右子树用'#'表示
void treeNodeConstructor(Tree &root, char data[]) {
    char e = data[index++];
    if(e == '#') {
        root = NULL;
    } else {
        root = (Node *)malloc(sizeof(Node));
        root->data = e;
        treeNodeConstructor(root->lchild, data);  //递归构建左子树
        treeNodeConstructor(root->rchild, data);  //递归构建右子树
    }
}
// DFS Depth first
void depthFirstSearch(Tree root) {
    stack<Node *> nodeStack;  //使用C++的STL标准模板库
    nodeStack.push(root);
    Node *node;
    while(!nodeStack.empty()) {
        node = nodeStack.top();
        cout << node->data;//遍历根结点
        nodeStack.pop();
        if(node->rchild) {
            nodeStack.push(node->rchild);  //先将右子树压栈
        }
        if(node->lchild) {
            nodeStack.push(node->lchild);  //再将左子树压栈
        }
    }
}

// BFS
void breadthFirstSearch(Tree root) {
    queue<Node *> nodeQueue;  //使用C++的STL标准模板库
    nodeQueue.push(root);
    Node *node;
    while(!nodeQueue.empty()) {
        node = nodeQueue.front();
        nodeQueue.pop();
        cout<<node->data;//遍历根结点
        if(node->lchild) {
            nodeQueue.push(node->lchild);  //先将左子树入队
        }
        if(node->rchild) {
            nodeQueue.push(node->rchild);  //再将右子树入队
        }
    }
}

void postorderTraversal(Tree root) {
    stack<Node *> treeNodeStack;
    Node *node = root;
    Tree lastVisit = root;
    while (node != NULL || !treeNodeStack.empty()) {
        while (node != NULL) {
            treeNodeStack.push(node);
            node = node->lchild;
        }
        //查看当前栈顶元素
        node = treeNodeStack.top();
        //如果其右子树也为空，或者右子树已经访问
        //则可以直接输出当前节点的值
        if (node->rchild == NULL || node->rchild == lastVisit) {
            cout << node->data;
            treeNodeStack.pop();
            lastVisit = node;
            node = NULL;
        } else {
            //否则，继续遍历右子树
            node = node->rchild;
        }
    }
}

void inorderTraversal(Tree root) {
    stack<Node *> s;
    Node *p = root;
    while(p != NULL || !s.empty()) {
        while(p != NULL) {
            s.push(p);
            p = p->lchild;
        }
        if(!s.empty()) {
            p = s.top();
            s.pop();
            cout << p->data;
            p = p->rchild;
        }
    }
}

void preorderTraversal(Tree root) {
    stack<Node *> s;
    Node *p = root;
    while(p != NULL || !s.empty()) {
        while(p != NULL) {
            cout << p->data;
            s.push(p);
            p = p->lchild;
        }
        if(!s.empty()) {
            p = s.top();
            s.pop();
            p = p->rchild;
        }
    }
}

int main() {
    //上图所示的二叉树先序遍历序列,其中用'#'表示结点无左子树或无右子树
    char data[15] = {'A', 'B', 'D', '#', '#', 'E', '#', '#', 'C', 'F','#', '#', 'G', '#', '#'};
    Tree tree;
    treeNodeConstructor(tree, data);
    printf("depthFirstSearch: ");
    depthFirstSearch(tree);
    printf("\nbreadthFirstSearch: ");
    breadthFirstSearch(tree);
    printf("\npostorderTraversal: ");
    postorderTraversal(tree);
    printf("\ninorderTraversal: ");
    inorderTraversal(tree);
    printf("\npreorderTracersal: ");
    preorderTraversal(tree);
    printf("\n");
    return 0;
}
