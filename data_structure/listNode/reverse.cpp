 #include <stdio.h>
 #include <stdlib.h>
 
 struct ListNode {
     int val;
     struct ListNode *next;
 };

#if 1
ListNode* reverse(ListNode* head) {
    ListNode* pre = NULL;
    ListNode* curr = head;
    while (curr != NULL) {
        ListNode* next = curr->next;
        curr->next = pre;
        pre = curr;
        curr = next;
    }
    return pre;
}
#else 
ListNode* reverse(ListNode* head) {
    ListNode* new_head = NULL;
    ListNode* curr = head;
    while (curr != NULL) {
        ListNode* next = curr->next;
        curr->next = new_head;
        new_head = curr;
        curr = next; // Save for next loop
    }
    return new_head;
}

#endif

ListNode* reverseKGroup(ListNode* head, int k) {
    ListNode* dummy = (ListNode*)malloc(sizeof(ListNode));
    dummy->next = head;

    ListNode* pre = dummy;
    ListNode* end = dummy;

    while (end->next != NULL) {
        for (int i = 0; i < k && end != NULL; i++) end = end->next;
        if (end == NULL) break;
        ListNode* start = pre->next;
        ListNode* next = end->next;
        end->next = NULL;
        pre->next = reverse(start);
        start->next = next;
        //pre = start;

        //end = pre;
        break; // Here can break I think
    }
    return dummy->next;
}



ListNode* createList(ListNode* head, int data) {
    ListNode *p = (ListNode*)malloc(sizeof(ListNode));
    p->val = data;
    p->next = NULL;
    
    if (head == NULL) {
        head = p;
        return head;
    }
    p->next = head;
    head = p;
    return head;
}

void printList(ListNode* head) {
    ListNode* p = head;
    while (p != NULL) {
        printf(" %d ", p->val);
        p = p->next;
    }
    printf("\n");
}

int main() {
    int i = 0;
    
    ListNode *head = NULL;
    for (i = 8; i > 0; i--) {
        head = createList(head, i);
    }
    
    printList(head);
    
   // head = reverse(head);
   // printList(head);
    head = reverseKGroup(head, 5);
    printList(head);
    
    return 0;
}
