
// ц╟ещеепР
void BubbleSort(vector<int>& v) {
    int len = v.size();
    for (int i = 0; i < len - 1; ++i)
        for (int j = 0; j < len - 1 - i; ++j)
	    if (v[j] > v[j + 1]) 
		swap(v[j], v[j + 1]);
}
